---
name: tb-pick-and-place
description: Full pick-and-place skill combining IBVS visual servoing pick with raise-and-look-around visual servoing place. Use when (1) the user asks to move an object from one place to another, (2) "pick up X and put it on Y", (3) any task requiring both grasping and placing.
---

# Pick and Place

Chains `tb-pick-up-object` (two-phase IBVS servo-descend) with `tb-place-object` (raise-and-look-around + mask-only tracking).

## Usage

```python
from main import pick_and_place
pick_and_place(pick_target="yellow banana", place_target="red plate")
```

## Pipeline

1. Init gripper, tilt camera, detect pick target (with search)
2. Servo-descend with XY+rotation search
3. Grasp, go home
4. Tilt camera, detect place target from height (mask-only)
5. Servo laterally above place target
6. Descend to place height while tracking
7. Release, go home

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| PICK_TARGET | "yellow banana" | Object to pick up |
| PLACE_TARGET | "red plate" | Where to place it |
| CAMERA_ID | "309622300814" | Wrist camera ID |
| EE_FRAME_Z_THRESHOLD | -0.25 | Switch from base to EE frame |
| PLACE_Z | -0.35 | Target place height (meters) |

## Dependencies

`tb-pick-up-object`, `tb-place-object` — see `scripts/deps.txt`. Bundle with `tidybot-bundle` before submission.
