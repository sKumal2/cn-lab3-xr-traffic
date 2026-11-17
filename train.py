"""Step III: train random forest classifiers on 5-second traffic windows.

    python train.py --sessions data/dummy/sessions.csv --out outputs/dummy
    python train.py --sessions data/sessions.csv       --out outputs/real

Tasks
  video        A vs B             (user 1 video sessions only, so user isn't a confound)
  application  idle/video/web     (all sessions)
  user         1 vs 2             (video A sessions only, so the video isn't a confound)
  case         one label per Step II session (optional, for the Step IV challenge)

How data leakage is prevented
  1. Chronological split within each session: the first part of every session
     is used for training and the last part for testing. Adjacent 5 s windows
     of one trace are strongly correlated, so a random shuffle would put
     near-duplicates on both sides and inflate accuracy.
  2. A purge gap of --gap windows is dropped between train and test so no
     burst spans the boundary.
  3. The first and last --trim windows of each session (start/stop
     artifacts) are dropped, so the start/stop of a session can't hint at its label.
  4. Features are only the four required fields. No IPs, ports, timestamps,
     window index, or session id ever reach the model.
  5. Hyperparameters are fixed up front. The test windows are used once, for
     the final report, never for fitting or tuning.
  6. Code asserts that train and test windows do not overlap.
After evaluation each model is refit on all windows and saved for Step IV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

import xr_traffic

RF_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

TASKS = {
    "video": dict(label="video", where=lambda s: (s.application == "video") & (s.user == "1")),
    "application": dict(label="application", where=lambda s: s.application.ne("")),
    "user": dict(label="user_label", where=lambda s: (s.application == "video") & (s.video == "A")),
    "case": dict(label="case", where=lambda s: s.case.ne("")),
}


def load_sessions(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, dtype=str).fillna("")
    s.columns = s.columns.str.strip()
    for c in ["session_id", "file", "case", "video", "application", "user"]:
        if c not in s:
            s[c] = ""
        s[c] = s[c].str.strip()
    s["user"] = s["user"].str.replace(r"\.0$", "", regex=True)
    if s.session_id.duplicated().any():
        raise SystemExit("session_id values in sessions.csv must be unique")
    return s


def build_windows(sessions: pd.DataFrame, base: Path) -> pd.DataFrame:
    frames = []
    for row in sessions.itertuples():
        if not (base / row.file).exists():
            print(f"  WARNING: {base / row.file} not found, skipping session {row.session_id}")
            continue
        ips = [ip.strip() for ip in row.local_ips.split(";") if ip.strip()] if "local_ips" in sessions else None
        start = float(row.start_s) if getattr(row, "start_s", "") else None
        end = float(row.end_s) if getattr(row, "end_s", "") else None
        w, used = xr_traffic.capture_to_windows(base / row.file, ips, start, end)
        print(f"  {row.session_id:<14} {len(w):>3} windows  local_ips={sorted(used)}")
        for c in ["session_id", "case", "video", "application", "user"]:
            w[c] = getattr(row, c)
        w["user_label"] = "user" + row.user  # text labels; "1"/"2" trip up sklearn
        frames.append(w)
    if not frames:
        raise SystemExit("no capture files found; check the file column in sessions.csv")
    return pd.concat(frames, ignore_index=True)


def chrono_split(df: pd.DataFrame, test_frac: float, gap: int, trim: int):
    """Per-session time split: [trim | train | gap | test | trim]."""
    tr, te = [], []
    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values("window")
        if trim:
            g = g.iloc[trim:-trim]
        n_test = max(1, int(round(len(g) * test_frac)))
        n_train = len(g) - n_test - gap
        if n_train < 2:
            raise SystemExit(f"session {g.session_id.iloc[0]} too short to split")
        tr.append(g.iloc[:n_train])
        te.append(g.iloc[n_train + gap:])
    train, test = pd.concat(tr), pd.concat(te)
    # leakage guard: no (session, window) pair is in both sets, and every
    # test window comes after every train window of the same session
    assert not set(zip(train.session_id, train.window)) & set(zip(test.session_id, test.window))
    last_train = train.groupby("session_id").window.max()
    first_test = test.groupby("session_id").window.min()
    assert (first_test > last_train + gap - 1).all()
    return train, test


def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 3.6 + 0.4 * len(labels)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train Step III random forest models")
    ap.add_argument("--sessions", default="data/sessions.csv")
    ap.add_argument("--out", default="outputs/real")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--gap", type=int, default=1, help="windows purged between train and test")
    ap.add_argument("--trim", type=int, default=1, help="windows dropped at each session edge")
    ap.add_argument("--tasks", default="video,application,user,case")
    args = ap.parse_args()

    sessions_path = Path(args.sessions)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    print(f"Reading captures listed in {sessions_path}")
    sessions = load_sessions(sessions_path)
    windows = build_windows(sessions, sessions_path.parent)
    windows.to_csv(out / "windows.csv", index=False)

    train_all, test_all = chrono_split(windows, args.test_frac, args.gap, args.trim)
    X_cols = xr_traffic.FEATURES
    report, summary = [], {}

    for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        spec = TASKS[name]
        keep = sessions.loc[spec["where"](sessions), "session_id"]
        tr = train_all[train_all.session_id.isin(keep)]
        te = test_all[test_all.session_id.isin(keep)]
        classes = sorted(tr[spec["label"]].unique())
        if len(classes) < 2:
            print(f"\n[{name}] skipped: needs >= 2 classes, found {classes}")
            continue

        model = RandomForestClassifier(**RF_PARAMS).fit(tr[X_cols], tr[spec["label"]])
        pred = model.predict(te[X_cols])
        acc = accuracy_score(te[spec["label"]], pred)
        cm = confusion_matrix(te[spec["label"]], pred, labels=classes)
        plot_cm(cm, classes, f"{name}  (test acc {acc:.2f}, n={len(te)})", out / f"cm_{name}.png")

        text = (
            f"=== {name}: {' vs '.join(classes)} ===\n"
            f"sessions: {', '.join(keep)}\n"
            f"train windows: {len(tr)}   test windows: {len(te)}   test accuracy: {acc:.3f}\n"
            f"confusion matrix (rows = true, cols = predicted) {classes}\n{cm}\n"
            f"{classification_report(te[spec['label']], pred, labels=classes, zero_division=0)}"
            f"feature importance: "
            + ", ".join(f"{f}={v:.2f}" for f, v in zip(X_cols, model.feature_importances_))
            + "\n"
        )
        print("\n" + text)
        report.append(text)

        # final model for Step IV: same hyperparameters, all labelled windows
        all_w = windows[windows.session_id.isin(keep)]
        final = RandomForestClassifier(**RF_PARAMS).fit(all_w[X_cols], all_w[spec["label"]])
        joblib.dump({"model": final, "features": X_cols, "window_s": xr_traffic.WINDOW_S,
                     "task": name, "classes": classes},
                    out / "models" / f"rf_{name}.joblib")
        summary[name] = {"classes": classes, "test_accuracy": round(acc, 4),
                         "n_train": len(tr), "n_test": len(te), "confusion_matrix": cm.tolist()}

    # one figure with the three required confusion matrices, for the report
    required = [t for t in ("video", "application", "user") if t in summary]
    if required:
        fig, axes = plt.subplots(1, len(required), figsize=(5 * len(required), 4.6))
        for ax, t in zip(np.atleast_1d(axes), required):
            s = summary[t]
            ConfusionMatrixDisplay(np.array(s["confusion_matrix"]), display_labels=s["classes"]).plot(
                ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{t}: test acc {s['test_accuracy']:.2f} (n={s['n_test']})")
        fig.suptitle("Random forest, 5 s windows, rows = true label, columns = predicted")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrices.png", dpi=150)
        plt.close(fig)

    missing = [t for t in ("video", "application", "user") if t not in summary]
    if missing:
        print(f"\nWARNING: required task(s) not trained: {missing} (missing sessions/classes)")
    apps = sorted(summary.get("application", {}).get("classes", []))
    if "application" in summary and apps != ["idle", "video", "web"]:
        print(f"\nWARNING: application classes are {apps}; the lab asks for idle/video/web. "
              "Record an idle capture.")

    header = (f"split: per-session chronological, test_frac={args.test_frac}, "
              f"gap={args.gap}, trim={args.trim}\nrf params: {RF_PARAMS}\n\n")
    (out / "report.txt").write_text(header + "\n".join(report))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved report, confusion matrices, windows.csv and models to {out}/")


if __name__ == "__main__":
    main()
