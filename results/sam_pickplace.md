# Pick-and-place with SAM masks (Isaac Sim 6.1, n = 10)

The robot picked 6 of 10 objects and carried all 6 to the detected basket. About 5 of 10 ended inside the
real basket.

## Setup

- Perception: Segment Anything (ViT-B) on the main camera, run on the CPU, with the table, the workspace
  and the robot's own pixels masked out. No ground-truth masks are used (`safegrasp/perception/sam_seg.py`).
- Target: the first visible, reachable object in each scene, matched to its SAM mask.
- Basket: placed at (0.58, -0.33) m, but located in every run by `detect_basket` from colour and depth
  (centre of the red bounding box). The drop point comes only from this detection.
- Method `ant_nom`, medium clutter, seeds 1 to 10.
- A place counts as successful when the object reaches within 0.11 m of the detected basket centre.
- Raw rows: `sam_pickplace.json`. Commands: `scripts/batch_pickplace.py` for the metrics and
  `scripts/compose_sam.py` for the four-panel video.

## Results

| seed | target | basket error [m] | picked | placed | failure |
|---|---|---|---|---|---|
| 1 | 11pro_SL_TRX_FG | 0.008 | yes | yes | |
| 2 | 2_of_Jenga_Classic_Game | 0.008 | yes | yes | |
| 3 | 11pro_SL_TRX_FG | 0.008 | yes | yes | |
| 4 | 2_of_Jenga_Classic_Game | 0.226 | no | no | slip |
| 5 | 11pro_SL_TRX_FG | 0.193 | yes | at the detected spot only | |
| 6 | 3D_Dollhouse_Lamp | 0.008 | yes | yes | |
| 7 | 3D_Dollhouse_Happy_Brother | 0.052 | no | no | slip |
| 8 | 30_CONSTRUCTION_SET | 0.008 | no | no | slip |
| 9 | 2_of_Jenga_Classic_Game | 0.008 | no | no | no_grasp |
| 10 | 3D_Dollhouse_Sink | 0.008 | yes | yes | |

In seed 5 the basket was partly hidden and the detection was 19 cm off, so the object was put down at
the detected spot and most likely missed the real basket.

| metric | value |
|---|---|
| pick (grasp and carry) | 6/10 |
| place (reach the detected basket) | 6 of the 6 picked objects |
| object inside the real basket | 5/10 (inferred, since only arrival at the detected centre was logged) |
| basket detection error | mean 5.3 cm, below 1 cm in 7/10, 19 and 23 cm in the two partly hidden cases |
| failures | 3 slips, 1 no grasp |

The pick rate is close to the 57-60% that the same method reached with ground-truth masks, so SAM's
masks cost little. Once an object was held, the place step never failed. The weak point is basket
detection when another object hides part of the basket. The grasp failures are the same slips seen in
the benchmark.
