# SpaceMouse Teleoperation Setup

## Context
- SpaceMouse Compact is physically connected to **Windows PC**
- Simulation runs on **Linux machine** (andres-CAPTIVA-PC)
- Connection is via SSH + WebRTC (Isaac Sim streaming)

## Every Time You Want to Use the SpaceMouse

### Step 1: Windows — Start usbipd server
Open PowerShell and run:
```
usbipd list
```
Find the SpaceMouse bus ID (e.g. `1-2`), then:
```
usbipd bind --busid 1-2
```
> Note: if already bound from last time, `bind` will say it's already bound — that's fine, just skip it.

### Step 2: Linux — Attach the device
```bash
sudo modprobe vhci-hcd
sudo usbip attach -r <windows-ip> -b 1-2
```
Replace `<windows-ip>` with your Windows machine IP (find it with `ipconfig` on Windows).

Verify it worked:
```bash
lsusb | grep -i 3dconnexion
```
Should show: `3Dconnexion SpaceMouse Compact`

### Step 3: Linux — Start teleoperation
```bash
cd ~/Documents
MasterProject/isaac-inference/remote.sh leisaac/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --teleop_device=spacemouse \
  --num_envs=1 \
  --device=cuda \
  --enable_cameras \
  --record \
  --dataset_file=./datasets/dataset_spacemouse.hdf5
```

Open browser at `http://localhost:8211/streaming/webrtc-demo` to see the simulation.

Press **B** to start control, **N** to save episode as success, **R** to discard.

---

## If the Device Disconnects Mid-Session
```bash
# Detach and reattach
sudo usbip detach -p 00  # port number shown in: sudo usbip port
sudo usbip attach -r <windows-ip> -b 1-2
```

## Troubleshooting
- **lsusb doesn't show SpaceMouse**: re-run `usbip attach` on Linux
- **usbip attach fails**: make sure `usbipd bind` was run on Windows first
- **SpaceMouse reads all zeros**: move it while the test script runs, wait 1-2 seconds for first event
- **Teleop script crashes on spacemouse init**: check `pip show pyspacemouse` and `sudo apt install libhidapi-hidraw0`