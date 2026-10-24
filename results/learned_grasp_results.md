# Learned grasp proposers and the wrist-camera re-plan (Isaac Sim 6.1 pilots)

These are 15-scene pilots: medium clutter, ground-truth masks, pick-and-lift, paired per scene so that
only the proposer or the execution differs. No script in this repository re-runs them, and their raw logs
were not kept. The analytic baseline `ant_nom` succeeded in 53-67% of the same scenes from run to run.

## GG-CNN2 (`method = ggcnn`)

GG-CNN2 predicts grasp quality, angle and width per pixel from a top-down depth image. Our cameras look
across the table, so it runs on a top-down heightmap built from the observed point cloud, with the quality
map restricted to the target (`safegrasp/grasping/ggcnn_grasp.py`, Cornell-pretrained weights).

| run | `ant_nom` | `ggcnn` | `ggcnn` failures |
|---|---|---|---|
| with the analytic pad filter | 9/15 | 4/15 | slip 8, approach stall 3 |
| without it (as for VGN) | 8/15 | 3/15 | slip 6, approach stall 6 |

It lost for three likely reasons. It predicts top-down grasps only, and the top-down baseline `td_nom`
already trailed the antipodal grasp in the benchmark (47% against 60%). It was trained on real depth but
receives a synthetic heightmap, so its quality peaks fell on edges and corners and its width output
saturated. Finally, even well-placed grasps slipped during the close and lift, as with every proposer.

## VGN (`method = vgn`)

VGN predicts a 6-DoF grasp per voxel from a 40^3 TSDF. We build the TSDF over a 30 cm window around the
target, keep peaks near it and convert them to grasps (`safegrasp/grasping/vgn_grasp.py`, pretrained
`vgn_conv.pth`). It succeeded in 0/15 scenes (12 slips, 1 IK failure, 1 transit stall, 1 transit
timeout) against 8/15 for `ant_nom`. Its grasps look reasonable on their own, so the conversion from
VGN's grasp frame to our gripper frame may be wrong. The report therefore leaves VGN out of its results.

## Wrist-camera re-plan (`TrialCfg(closed_loop=True)`)

At the pre-grasp pose the robot captures a wrist camera image (`scene.wrist_capture`), which sees the
target closely (about 21k pixels, centroid within about 1.3 cm of ground truth). The wrist points are
fused with the main-camera cloud and the grasp is planned again.

- Always using the new grasp halved success (10/15 to 5/15), because good grasps were often replaced by
  worse ones.
- Switching only when the new grasp holds at least 20% more target points between the fingers gave 8/15
  against 10/15. It fixed three slips (seeds 3, 10 and 12) but broke five successes (1, 2, 5, 11 and 13).

The gated version cannot pick a worse grasp, yet it still lost, because the wrist capture itself steps
the physics. A re-planned trial therefore drifts from its open-loop twin even when it keeps the same
grasp.

## Summary

No proposer beat the plain antipodal grasp, and the re-plan did not help either. Earlier tries (a
stability re-rank, adaptive width, lift dynamics, friction, camera and centring changes) were also
negative. Most of the remaining failures are slips during the close on hard objects (thin Jenga blocks,
a phone, a shoe), together with CPU PhysX contacts that are not repeatable across scene restores. A
higher success rate likely needs better contact physics, such as GPU PhysX, rather than changes to the
pipeline.
