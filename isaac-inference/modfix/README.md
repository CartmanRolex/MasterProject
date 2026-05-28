# Modfixes

Portable fixes for source checkouts that live outside this repository.

## LeIsaac Camera Randomization Baseline

The patch `leisaac_camera_randomization_baseline.patch` fixes camera pose
randomization compounding into a random walk on machines where Isaac refreshes
`asset.data.pos_w` to the previous randomized camera pose before the next reset.

Apply it from the LeIsaac repository root:

```bash
cd ~/Documents/leisaac
git apply ~/Documents/MasterProject/isaac-inference/modfix/leisaac_camera_randomization_baseline.patch
```

Then run the drift diagnostic from this repository:

```bash
cd ~/Documents/MasterProject/isaac-inference
./remote.sh debug_camera_drift.py
```

After the fix, `REFERENCE` should remain constant across episodes.
