# pick-and-place

Author: evilsky
Dependencies: pick-up-object, place-object

Full pick-and-place skill combining IBVS visual servoing pick (from `pick-up-object`) with raise-and-look-around visual servoing place (from `place-object`).

Key features:
- **Pick**: Two-phase IBVS servo-descend (base frame → EE frame), XY + rotation search wiggle on detection miss, gradual gripper offset transition, ArmError-based floor detection
- **Place**: Raise to home height (camera sees past held object), mask-only detection for robust tracking, servo above target then descend while tracking
- **Configurable targets**: Pick and place targets specified as text prompts for YOLO open-vocabulary detection

## Usage

```python
from main import pick_and_place
pick_and_place(pick_target="yellow banana", place_target="red plate")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| PICK_TARGET | "yellow banana" | Object to pick up |
| PLACE_TARGET | "red plate" | Where to place it |
| CAMERA_ID | "309622300814" | Wrist camera ID |
| EE_FRAME_Z_THRESHOLD | -0.25 | Switch from base to EE frame |
| PLACE_Z | -0.35 | Target place height (meters) |

## Pipeline

1. Init gripper, tilt camera, detect pick target (with search)
2. Servo-descend with XY+rotation search
3. Grasp, go home
4. Tilt camera, detect place target from height (mask-only)
5. Servo laterally above place target
6. Descend to place height while tracking
7. Release, go home

## Components

- Pick logic from [pick-up-object](https://github.com/Tidybot-Skills/pick-up-object)
- Place logic from [place-object](https://github.com/Tidybot-Skills/place-object)
