"""Step IV: label every 5 s window of an unlabelled capture with saved models.

    python predict.py other_team.pcapng --models outputs/real/models
    python predict.py other_team.pcapng --models outputs/real/models --local-ip 10.0.0.23

Writes <capture>_predictions.csv: one row per window with its time range,
the four features, and each model's prediction plus confidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

import xr_traffic


def main():
    ap = argparse.ArgumentParser(description="Predict labels per 5 s window")
    ap.add_argument("capture")
    ap.add_argument("--models", default="outputs/real/models")
    ap.add_argument("--local-ip", action="append",
                    help="laptop IP in the capture (repeat for IPv4 + IPv6); auto-detected if omitted")
    ap.add_argument("--out")
    args = ap.parse_args()

    bundles = [joblib.load(p) for p in sorted(Path(args.models).glob("rf_*.joblib"))]
    if not bundles:
        raise SystemExit(f"no rf_*.joblib models in {args.models}; run train.py first")

    windows, ips = xr_traffic.capture_to_windows(args.capture, args.local_ip)
    print(f"{len(windows)} windows, local_ips={sorted(ips)}")
    w = bundles[0]["window_s"]
    windows.insert(1, "start_s", windows["window"] * w)
    windows.insert(2, "end_s", (windows["window"] + 1) * w)

    for b in bundles:
        X = windows[b["features"]]
        windows[f"pred_{b['task']}"] = b["model"].predict(X)
        windows[f"conf_{b['task']}"] = b["model"].predict_proba(X).max(axis=1).round(3)

    out = args.out or str(Path(args.capture).with_suffix("")) + "_predictions.csv"
    windows.to_csv(out, index=False)
    print(windows.drop(columns=xr_traffic.FEATURES).to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
