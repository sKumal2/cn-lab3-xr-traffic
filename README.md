# Lab 3 (HMD): Step III traffic classifier

Random forest classifiers that predict what an XR session is doing from
5-second windows of Wireshark traffic.

## Setup
```sh
source .venv/bin/activate        # .venv links to ~/.venvs/cn-lab3
pip install -r requirements.txt  # only needed on a new machine
```

## Data
Captures are in `data/raw/video_packets/` and labelled in `data/sessions.csv`.
The laptop IP in all captures is `10.200.60.15`.

| File | Session | App | Video | User | Length | Windows |
|---|---|---|---|---|---|---|
| `Video1.pcapng` | video A | DeoVR | Beyond a Mask | Linda | 208 s | 41 |
| `Video2.pcapng` | video B | DeoVR | Dolphin Immersion | Linda | 180 s (labelled from 5 s) | 34 |
| `WebVideo.pcapng` | web | YouTube (browser) | – | James | 33 s | 6 |

By team decision, the idle capture and the second user watching video A were
skipped.

## How it works
1. `train.py` reads `data/sessions.csv` (file path + labels for each capture).
2. `xr_traffic.py` turns each trace into 5-second windows with four fields:
   `download_bytes`, `upload_bytes`, `packet_count`, `remote_endpoints`.
   Only traffic to/from the laptop is counted. All windows are saved to
   `outputs/real/windows.csv`.
3. A random forest is trained per task (300 trees, `min_samples_leaf=2`,
   balanced class weights, `random_state=42`) and evaluated on held-out windows.
4. Confusion matrices, a report, and the trained models are saved to `outputs/real/`.

## Results
Test set = the last ~30% of each session's windows (see data-leakage controls).

**Video A vs B**: 22 / 22 correct, accuracy 1.00
```
              pred A  pred B
true A          12       0
true B           0      10
```
Most important features: download bytes (0.63), packet count (0.33).
Video A streams at about 25 MB per window (~40 Mbps) and video B at about
31–34 MB (~50 Mbps); the two ranges do not overlap, so bitrate alone
separates them.

**Application, video vs web**: 23 / 23 correct, accuracy 1.00
```
              pred video  pred web
true video        22          0
true web           0          1
```
The web capture is only 33 s, so this rests on a single web test window
and is weak evidence.

**User 1 vs user 2**: not trained (no second-user capture).

Extra **session** classifier (video A / video B / web, for Step IV):
23 / 23 correct.

Figure for the report: `outputs/real/confusion_matrices.png`.
Full numbers: `outputs/real/report.txt` and `outputs/real/summary.json`.

## Why this matters: traffic analysis
The goal of the lab is to show what an attacker can learn about an XR user
just by **watching** network packets. This is called traffic analysis. It is
passive: the attacker doesn't modify or block anything, and doesn't break
encryption.

DeoVR and YouTube traffic is encrypted, so an eavesdropper can't see the
video, the page or anything typed. They can still see the size and timing of
each packet, the number of packets, and which servers are contacted. Those are
exactly the four features our model uses. It never looked inside a packet.

**What our results show**

| Result | What it means for privacy |
|---|---|
| Video A vs B: 22 / 22 | Someone on the same network can tell **which video** a user is watching, even though it's encrypted. Each video has its own traffic pattern (~40 vs ~50 Mbps here). |
| Video vs web: 23 / 23 | They can tell **what the user is doing**: VR video or web browsing. |
| One server (`89.187.173.87`) carried 99.9% of the video traffic | The server address alone reveals which service is in use. |

With more training captures the same approach goes further. Research has
used it to identify which video title or website someone is viewing, and in
VR to identify **who the user is** from the head and hand motion data sent
upstream, since every person moves differently. That is what the skipped
user 1 vs user 2 classifier would have tested.

**Who could do this:** anyone on the same public WiFi (like the GSU network we
used), whoever runs the router or network (ISP, employer, school), or someone
who has compromised a router.

**Why XR makes it worse:** headsets send large, continuous streams of data,
including movement, voice and room scans. The traffic pattern can reveal not
just what someone watches but things about their body and behaviour.

**Defences**
- **VPN:** hides which servers are contacted, but not packet sizes and
  timing, so the video A vs B model would still work.
- **Padding / constant-rate sending:** makes every session look the same.
  Effective, but it costs bandwidth.
- **Traffic shaping, added noise, or fake traffic mixed in:** blurs the
  patterns the model relies on.

**Summary:** using only packet sizes, counts and endpoints from encrypted
traffic, a simple random forest identified which VR video we were watching
with 100% accuracy on our test windows, so a passive eavesdropper on the same
WiFi could do the same. This is based on a small number of captures (one per
session).

## Run
```sh
python train.py --sessions data/sessions.csv --out outputs/real   # real captures
python predict.py <capture.pcapng> --models outputs/real/models   # Step IV
python make_dummy_data.py                                         # fake data → data/dummy/
python train.py --sessions data/dummy/sessions.csv --out outputs/dummy
```
`train.py` prints two warnings on the real data (no user model, no idle
class); these are expected given the skipped captures.

In Step IV, only trust the `video` prediction for windows that the
`application` model labels as video.

## Data-leakage controls
- Per-session chronological split: earlier windows train, later windows test.
  No random shuffle, because neighbouring windows are near-copies.
- One window is dropped between the train and test parts of each session.
- The first and last window of each session are dropped (start/stop effects).
- Only the four required features are used; no IPs, ports, timestamps or IDs.
- Hyperparameters are fixed in advance; the test set is used once.
- Video A vs B uses only one user's sessions, so user can't be a shortcut.
- The code asserts no window appears in both train and test.
- After evaluation, final models are refit on all windows and saved for Step IV.
