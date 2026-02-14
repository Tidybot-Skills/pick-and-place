"""Pick and place skill.

Pick phase: from pick_object2 (wishlist skill) — IBVS servo-descend with
  search wiggle, gradual gripper offset, mask orientation alignment.

Place phase: from pick_and_place — raise to home, tilt camera, live visual
  servo above place target, descend while tracking, release.

Workflow:
  1. Init gripper, tilt camera, detect pick target (with search)
  2. Servo-descend with XY+rotation search, align gripper at EE threshold
  3. Grasp, go home (holding object)
  4. Tilt camera, detect place target (camera sees past held object from height)
  5. Servo laterally above place target (live detection)
  6. Descend to place height while tracking
  7. Release, go home
"""

from robot_sdk import arm, gripper, sensors, yolo, display
from robot_sdk.arm import ArmError
import numpy as np
import time
import math

# ============================================================================
# Configuration
# ============================================================================

PICK_TARGET = "yellow banana"
PLACE_TARGET = "red plate"

CAMERA_ID = "309622300814"
DETECTION_CONFIDENCE = 0.15

# --- Visual servoing gains ---
GAIN_U_TO_DY = -0.0006
GAIN_U_TO_DX = 0.0
GAIN_V_TO_DX = -0.0006
GAIN_V_TO_DZ = 0.0

# --- Gripper-to-camera offset ---
GRIPPER_U_OFFSET = 0.0
GRIPPER_V_OFFSET = -120  # pixels

# --- Gradual gripper offset transition (pick only) ---
OFFSET_START_Z = 0.0
OFFSET_END_Z = -0.25

# --- Servoing loop parameters ---
PIXEL_TOLERANCE = 30
MAX_SERVO_ITERATIONS = 200
MAX_LATERAL_STEP_M = 0.05
MIN_LATERAL_STEP_M = 0.001
SERVO_MOVE_DURATION = 0.5

# --- Search parameters (pick) ---
SEARCH_WIGGLE_ANGLE_DEG = 30
SEARCH_XY_STEP_M = 0.05
SEARCH_WIGGLE_DURATION = 0.4
MAX_SEARCH_FAILURES = 3

# --- Descent parameters ---
DESCEND_STEP_M = 0.05
DESCEND_PAUSE_PIXELS = 80

# --- EE frame mode (pick only) ---
EE_FRAME_Z_THRESHOLD = -0.25  # Same as OFFSET_END_Z
EE_GAIN_U_TO_DY = +1.0 / 580
EE_GAIN_V_TO_DX = -1.0 / 660

# --- Grasp parameters ---
GRASP_FORCE = 50
GRASP_SPEED = 200

# --- Place parameters ---
PLACE_Z = -0.35
PLACE_DESCEND_STEP_M = 0.03
PLACE_PIXEL_TOLERANCE = 40
PLACE_LOST_RETRIES = 5

# --- Camera tilt ---
CAMERA_TILT_RAD = math.radians(-20)


# ============================================================================
# Helper functions
# ============================================================================

def detect_object_2d(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target object with mask and return the best detection."""
    result = yolo.segment_camera(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    best = max(detections, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def detect_object_mask_only(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target object, only return detections that have a valid mask.

    For place phase: as long as the mask is visible, we have the target.
    Ignores bbox-only detections.
    """
    result = yolo.segment_camera(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    # Filter to mask-only detections
    masked = [d for d in detections if d.mask is not None and (d.mask > 0.5).sum() > 0]
    if not masked:
        return None, None
    best = max(masked, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def get_object_pixel_center(detection):
    """Get object center in pixel coordinates (mask centroid or bbox center)."""
    if detection.mask is not None:
        mask = detection.mask
        binary = (mask > 0.5).astype(np.float32)
        total = binary.sum()
        if total > 0:
            ys, xs = np.where(binary > 0)
            return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_mask_orientation(detection):
    """Get object orientation angle from mask using PCA.

    Returns angle in radians for EE yaw rotation to align gripper PERPENDICULAR
    to object's long axis. Clamped to +/-90 deg. Returns 0 if no mask.
    """
    if detection.mask is None:
        return 0.0

    mask = detection.mask
    binary = (mask > 0.5).astype(np.float32)
    ys, xs = np.where(binary > 0)

    if len(xs) < 10:
        return 0.0

    cx, cy = xs.mean(), ys.mean()
    xs_c = xs - cx
    ys_c = ys - cy

    cov_xx = np.mean(xs_c * xs_c)
    cov_yy = np.mean(ys_c * ys_c)
    cov_xy = np.mean(xs_c * ys_c)

    theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

    # Add 90 deg to get perpendicular (gripper grabs across, not along)
    theta_perp = theta + math.pi / 2

    while theta_perp > math.pi:
        theta_perp -= 2 * math.pi
    while theta_perp < -math.pi:
        theta_perp += 2 * math.pi

    max_angle = math.pi / 2
    theta_perp = max(-max_angle, min(max_angle, theta_perp))

    # Negate for image-to-EE coordinate conversion
    return -theta_perp


def get_servo_target_pixel(image_shape, ee_z: float):
    """Return servo target pixel with gradual gripper offset (for pick).

    Offset interpolates from 0% at OFFSET_START_Z to 100% at OFFSET_END_Z.
    """
    h, w = image_shape[0], image_shape[1]
    if ee_z >= OFFSET_START_Z:
        ratio = 0.0
    elif ee_z <= OFFSET_END_Z:
        ratio = 1.0
    else:
        ratio = (OFFSET_START_Z - ee_z) / (OFFSET_START_Z - OFFSET_END_Z)
    u_offset = GRIPPER_U_OFFSET * ratio
    v_offset = GRIPPER_V_OFFSET * ratio
    return w / 2.0 + u_offset, h / 2.0 + v_offset


def get_image_center(image_shape):
    """Return image center pixel (for place — no gripper offset)."""
    h, w = image_shape[0], image_shape[1]
    return w / 2.0, h / 2.0


def pixel_error_to_ee_delta(u_err, v_err):
    """Convert pixel error to EE delta in base frame."""
    dx = GAIN_U_TO_DX * u_err + GAIN_V_TO_DX * v_err
    dy = GAIN_U_TO_DY * u_err
    dz = GAIN_V_TO_DZ * v_err
    step_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if step_norm > MAX_LATERAL_STEP_M:
        scale = MAX_LATERAL_STEP_M / step_norm
        dx *= scale
        dy *= scale
        dz *= scale
    return dx, dy, dz


# ============================================================================
# Search functions (pick phase)
# ============================================================================

def search_xy_sweep(target: str, axis: str):
    """Sweep +/-5cm in X or Y direction. Stays at found position."""
    step = SEARCH_XY_STEP_M
    if axis == 'x':
        arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    else:
        arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    det, shape = detect_object_2d(target)
    if det is not None:
        print(f"      Found at +{step*100:.0f}cm {axis.upper()} - staying here")
        return det, shape, step

    if axis == 'x':
        arm.move_delta(dx=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    else:
        arm.move_delta(dy=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    det, shape = detect_object_2d(target)
    if det is not None:
        print(f"      Found at -{step*100:.0f}cm {axis.upper()} - staying here")
        return det, shape, -step

    if axis == 'x':
        arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    else:
        arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    return None, None, 0


def search_at_current_rotation(target: str):
    """Search at current rotation by sweeping XY."""
    det, shape = detect_object_2d(target)
    if det is not None:
        return det, shape
    print(f"      Sweeping X +/-{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape, _ = search_xy_sweep(target, 'x')
    if det is not None:
        return det, shape
    print(f"      Sweeping Y +/-{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape, _ = search_xy_sweep(target, 'y')
    if det is not None:
        return det, shape
    return None, None


def search_wiggle(target: str):
    """Search by rotating +/-30 deg and sweeping XY at each angle."""
    wiggle_rad = math.radians(SEARCH_WIGGLE_ANGLE_DEG)

    print(f"    Wiggle search: checking center with XY sweep...")
    det, shape = search_at_current_rotation(target)
    if det is not None:
        return det, shape, False

    print(f"    Wiggle search: rotating +{SEARCH_WIGGLE_ANGLE_DEG} deg...")
    arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found at +{SEARCH_WIGGLE_ANGLE_DEG} deg - descending & returning rotation")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass
        arm.move_delta(dyaw=-wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True

    print(f"    Wiggle search: rotating to -{SEARCH_WIGGLE_ANGLE_DEG} deg...")
    arm.move_delta(dyaw=-2*wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found at -{SEARCH_WIGGLE_ANGLE_DEG} deg - descending & returning rotation")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass
        arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True

    print(f"    Object not found, returning rotation to center...")
    arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    return None, None, False


# ============================================================================
# Pick phase — servo descend with search + orientation alignment
# ============================================================================

def servo_descend(target: str):
    """Servo-descend with XY+rotation search on detection miss.

    Two phases:
      - Base frame (Z > EE threshold): target = image center (gradual offset)
      - EE frame (Z < EE threshold): target = gripper offset, align to object once
    """
    ee_x, ee_y, ee_z = sensors.get_ee_position()
    consecutive_search_failures = 0
    aligned_to_object = False

    print(f"\n--- Servo-Descend: approaching '{target}' ---")
    print(f"  Current EE Z: {ee_z:.3f}m, descend step: {DESCEND_STEP_M*1000:.0f}mm")
    print(f"  EE frame switch at Z < {EE_FRAME_Z_THRESHOLD}m")
    display.show_text(f"Approaching {target}...")
    display.show_face("thinking")

    for i in range(MAX_SERVO_ITERATIONS):
        ee_x, ee_y, ee_z = sensors.get_ee_position()
        use_ee_frame = ee_z < EE_FRAME_Z_THRESHOLD

        # Orientation alignment disabled — causes unexpected 90° rotation
        # on round objects (cup, etc.) where PCA gives arbitrary axis.
        # if use_ee_frame and not aligned_to_object:
        #     ...

        det, shape = detect_object_2d(target)

        if det is None:
            print(f"  Iter {i+1}: object not detected, searching...")
            det, shape, found_at_wiggle = search_wiggle(target)
            if det is None:
                consecutive_search_failures += 1
                print(f"    Search failed ({consecutive_search_failures}/{MAX_SEARCH_FAILURES})")
                if consecutive_search_failures >= MAX_SEARCH_FAILURES:
                    print("  ERROR: Lost object after max search attempts.")
                    return False
                continue
            else:
                consecutive_search_failures = 0
                if found_at_wiggle:
                    continue
        else:
            consecutive_search_failures = 0

        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_servo_target_pixel(shape, ee_z)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        src = "mask" if det.mask is not None else "bbox"
        frame_str = "EE" if use_ee_frame else "BASE"
        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[{src}] [{frame_str}] Z={ee_z:.3f}m")

        if use_ee_frame:
            dx_lat = EE_GAIN_V_TO_DX * v_err
            dy_lat = EE_GAIN_U_TO_DY * u_err
            lat_norm = np.sqrt(dx_lat**2 + dy_lat**2)
            if lat_norm > MAX_LATERAL_STEP_M:
                scale = MAX_LATERAL_STEP_M / lat_norm
                dx_lat *= scale
                dy_lat *= scale
        else:
            dx_lat, dy_lat, _ = pixel_error_to_ee_delta(u_err, v_err)

        descend_this_step = 0.0
        if error_mag < DESCEND_PAUSE_PIXELS:
            descend_this_step = DESCEND_STEP_M
        else:
            print(f"    Pausing descent (error {error_mag:.0f} > {DESCEND_PAUSE_PIXELS}px)")

        dx = dx_lat
        dy = dy_lat
        dz = descend_this_step if use_ee_frame else -descend_this_step

        if np.sqrt(dx**2 + dy**2 + dz**2) < MIN_LATERAL_STEP_M:
            dz = DESCEND_STEP_M if use_ee_frame else -DESCEND_STEP_M
            descend_this_step = DESCEND_STEP_M
            dx, dy = 0.0, 0.0

        desc_str = f" down={descend_this_step*1000:.0f}mm" if descend_this_step > 0 else ""
        print(f"    Move [{frame_str}]: dx={dx*1000:.1f} dy={dy*1000:.1f} dz={dz*1000:.1f}{desc_str}")

        frame = "ee" if use_ee_frame else "base"
        try:
            arm.move_delta(dx=dx, dy=dy, dz=dz, frame=frame, duration=SERVO_MOVE_DURATION)
        except ArmError as e:
            print(f"  FLOOR CONTACT: {e}")
            print(f"  Final EE Z: {sensors.get_ee_position()[2]:.3f}m")
            return True

        time.sleep(0.2)

    print(f"  WARNING: Max iterations ({MAX_SERVO_ITERATIONS}) reached")
    return True


# ============================================================================
# Place phase — servo above target then descend
# ============================================================================

def servo_above_place(target: str):
    """Servo laterally to center above the place target.

    Uses image center as target (no gripper offset) since we're high up.
    Only corrects lateral error, no descent.
    Mask-only detection: only tracks if mask is visible (ignores bbox-only).
    """
    print(f"\n--- Servo-Above: centering over '{target}' (mask-only) ---")
    display.show_text(f"Centering over {target}...")
    display.show_face("thinking")

    consecutive_misses = 0

    for i in range(MAX_SERVO_ITERATIONS):
        det, shape = detect_object_mask_only(target)

        if det is None:
            consecutive_misses += 1
            print(f"  Iter {i+1}: mask not detected "
                  f"({consecutive_misses}/{PLACE_LOST_RETRIES})")
            if consecutive_misses >= PLACE_LOST_RETRIES:
                print("  ERROR: Lost place target mask. Aborting place.")
                return False
            time.sleep(0.5)
            continue
        consecutive_misses = 0

        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_image_center(shape)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        ee_z = sensors.get_ee_position()[2]
        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[mask] Z={ee_z:.3f}m")

        if error_mag < PLACE_PIXEL_TOLERANCE:
            print(f"  Centered above target!")
            return True

        dx, dy, _ = pixel_error_to_ee_delta(u_err, v_err)

        step = np.sqrt(dx**2 + dy**2)
        if step < MIN_LATERAL_STEP_M:
            print(f"    Centered (step too small). Done!")
            return True

        print(f"    Move: dx={dx*1000:.1f}mm, dy={dy*1000:.1f}mm")
        arm.move_delta(dx=dx, dy=dy, dz=0, droll=0, dpitch=0, dyaw=0,
                       frame="base", duration=SERVO_MOVE_DURATION)
        time.sleep(0.2)

    print(f"  WARNING: Max iterations reached.")
    return True


def descend_to_place(target: str):
    """Descend while keeping centered on place target.

    Mask-only detection: as long as mask is visible, keep tracking.
    """
    ee_x, ee_y, ee_z = sensors.get_ee_position()
    print(f"\n--- Descend-to-Place: going to Z={PLACE_Z:.3f}m (mask-only) ---")
    print(f"  Current EE Z: {ee_z:.3f}m")

    consecutive_misses = 0
    PLACE_HEIGHT_THRESHOLD = 0.02

    for i in range(MAX_SERVO_ITERATIONS):
        ee_x, ee_y, ee_z = sensors.get_ee_position()
        remaining = ee_z - PLACE_Z

        if remaining <= PLACE_HEIGHT_THRESHOLD:
            print(f"  Reached place height (Z={ee_z:.3f}m). Ready to release!")
            return True

        det, shape = detect_object_mask_only(target)

        if det is None:
            consecutive_misses += 1
            print(f"  Iter {i+1}: mask not detected "
                  f"({consecutive_misses}/{PLACE_LOST_RETRIES})")
            if consecutive_misses >= PLACE_LOST_RETRIES:
                print("  WARNING: Lost target mask, placing at current position.")
                return True
            time.sleep(0.5)
            continue
        consecutive_misses = 0

        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_image_center(shape)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[mask] Z={ee_z:.3f}m remain={remaining*100:.1f}cm")

        dx_lat, dy_lat, _ = pixel_error_to_ee_delta(u_err, v_err)

        descend_this_step = 0.0
        if error_mag < DESCEND_PAUSE_PIXELS:
            descend_this_step = min(PLACE_DESCEND_STEP_M, max(remaining, 0))
        else:
            print(f"    Pausing descent (error {error_mag:.0f} > "
                  f"{DESCEND_PAUSE_PIXELS}px), centering first...")

        dx = dx_lat
        dy = dy_lat
        dz = -descend_this_step

        total_step = np.sqrt(dx**2 + dy**2 + dz**2)
        if total_step < MIN_LATERAL_STEP_M and descend_this_step == 0:
            dz = -min(PLACE_DESCEND_STEP_M, max(remaining, 0))
            dx, dy = 0.0, 0.0

        print(f"    Move: dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm dz={dz*1000:.1f}mm")

        try:
            arm.move_delta(dx=dx, dy=dy, dz=dz, droll=0, dpitch=0, dyaw=0,
                           frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError as e:
            print(f"  CONTACT: {e}")
            return True

        time.sleep(0.2)

    print(f"  WARNING: Max iterations reached. Placing here.")
    return True


# ============================================================================
# Main pick-and-place pipeline
# ============================================================================

def pick_and_place(pick_target: str = PICK_TARGET, place_target: str = PLACE_TARGET):
    """Full pick-and-place pipeline.

    Pick: tilt camera, detect (with search), servo-descend with orientation
          alignment, grasp, go home.
    Place: tilt camera, detect place target (live), servo above, descend
           while tracking, release, go home.
    """
    print(f"=== Pick '{pick_target}' and Place on '{place_target}' ===\n")

    # --- Phase 0: Initialize ---
    print("Phase 0: Initializing...")
    display.show_text(f"Pick {pick_target} -> {place_target}")
    display.show_face("thinking")
    gripper.activate()
    gripper.open()
    time.sleep(0.5)

    # Tilt camera
    print("Tilting camera down...")
    arm.move_delta(dpitch=CAMERA_TILT_RAD, frame="ee", duration=1.0)
    time.sleep(0.3)

    # --- Phase 1: Detect pick target ---
    print("\nPhase 1: Detecting pick target...")
    det, shape = detect_object_2d(pick_target)
    if det is None:
        print("  Not detected, trying search...")
        det, shape, _ = search_wiggle(pick_target)
    if det is None:
        print(f"ERROR: '{pick_target}' not found. Aborting.")
        display.show_text(f"{pick_target} not found!")
        display.show_face("sad")
        return False

    obj_u, obj_v = get_object_pixel_center(det)
    src = "mask" if det.mask is not None else "bbox"
    print(f"  Detected at pixel ({obj_u:.0f}, {obj_v:.0f}) [{src}]")

    # --- Phase 2: Servo-descend to pick ---
    print("\nPhase 2: Servo-descend...")
    display.show_text(f"Picking {pick_target}...")
    reached = servo_descend(pick_target)
    if not reached:
        print("WARNING: Servo-descend incomplete, attempting grasp anyway.")

    # --- Phase 3: Grasp ---
    print("\nPhase 3: Grasping...")
    display.show_text(f"Grasping {pick_target}...")
    grasped = gripper.grasp(speed=GRASP_SPEED, force=GRASP_FORCE)
    time.sleep(0.5)

    if grasped:
        print("  Object grasped!")
        display.show_face("happy")
    else:
        print("  WARNING: No object detected in gripper.")
        display.show_face("concerned")

    # --- Phase 4: Go home (holding object) ---
    print("\nPhase 4: Going home (holding object)...")
    arm.go_home()
    time.sleep(0.5)

    # --- Phase 5: Tilt camera and find place target ---
    print(f"\nPhase 5: Looking for place target '{place_target}'...")
    display.show_text(f"Finding {place_target}...")

    # Tilt camera down — from home height, camera sees past held object
    arm.move_delta(dpitch=CAMERA_TILT_RAD, frame="ee", duration=1.0)
    time.sleep(0.3)

    det, shape = detect_object_mask_only(place_target)
    if det is None:
        print(f"  Place target mask not visible, aborting place.")
        print(f"  Opening gripper to drop object...")
        gripper.open()
        time.sleep(0.5)
        arm.go_home()
        display.show_face("sad")
        return False

    obj_u, obj_v = get_object_pixel_center(det)
    print(f"  Place target detected at pixel ({obj_u:.0f}, {obj_v:.0f}) [mask]")

    # --- Phase 6: Servo above place target ---
    print("\nPhase 6: Centering above place target...")
    display.show_text(f"Moving above {place_target}...")
    centered = servo_above_place(place_target)
    if not centered:
        print("WARNING: Could not center, placing at best position.")

    # --- Phase 7: Descend to place height ---
    print("\nPhase 7: Descending to place height...")
    display.show_text(f"Lowering to {place_target}...")
    descend_to_place(place_target)

    # --- Phase 8: Release ---
    print("\nPhase 8: Releasing object...")
    display.show_text("Releasing...")
    gripper.open()
    time.sleep(0.5)
    print("  Object released!")
    display.show_face("happy")
    display.show_text(f"Placed on {place_target}!")

    # --- Phase 9: Go home ---
    print("\nPhase 9: Going home...")
    arm.go_home()
    time.sleep(0.5)

    print(f"\n=== Pick and place complete! ===")
    display.show_face("excited")
    return True


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__" or True:
    success = pick_and_place(PICK_TARGET, PLACE_TARGET)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
