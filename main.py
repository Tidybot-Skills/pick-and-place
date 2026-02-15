"""Pick and place skill.

Chains pick-up-object and place-object to pick a target object and
place it on another target.

Dependencies (resolved by tidybot-bundle):
  - pick-up-object: IBVS visual servoing pick with servo-descend
  - place-object: Visual servoing place with raise-and-look-around

Usage:
  tidybot-bundle pick-and-place -o bundled.py
  # Then submit bundled.py via /code/execute
"""

from robot_sdk import display

# These functions are provided by dependencies (inlined by tidybot-bundle):
#   pick_up_object()  — from pick-up-object/main.py
#   place_object()    — from place-object/main.py

PICK_TARGET = "yellow banana"
PLACE_TARGET = "red plate"


def pick_and_place(pick_target=PICK_TARGET, place_target=PLACE_TARGET):
    """Pick up an object and place it on a target.

    Chains:
      1. pick-up-object: Find, servo-descend, grasp, go home (holding)
      2. place-object: Find place target from height, servo above, descend, release

    Args:
        pick_target: Object class name to pick up
        place_target: Object class name to place on

    Returns:
        bool: True if pick and place succeeded
    """
    print(f"=== Pick '{pick_target}' and Place on '{place_target}' ===\n")

    # Step 1: Pick up the object
    print("[STEP 1] Picking up object...")
    grasped = pick_up_object(target=pick_target)

    if not grasped:
        print(f"FAILED: Could not pick up '{pick_target}'")
        display.show_text(f"Failed to pick {pick_target}")
        display.show_face("sad")
        return False

    print(f"\nObject picked up successfully!")

    # Step 2: Place it on the target
    print("\n[STEP 2] Placing object...")
    placed = place_object(target=place_target)

    if placed:
        print(f"\n=== Pick and place complete! ===")
        display.show_face("excited")
    else:
        print(f"\n=== Place failed ===")

    return placed


if __name__ == "__main__":
    success = pick_and_place(PICK_TARGET, PLACE_TARGET)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
