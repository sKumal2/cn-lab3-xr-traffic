"""Generate synthetic .pcap captures that mimic the Step II sessions.

Lets us build and test the whole Step III pipeline before the real
Wireshark captures exist. Output:
    data/dummy/*.pcap
    data/dummy/sessions.csv

Each capture has ~5 s of unlabelled lead-in and lead-out traffic around the
labelled 180 s, so the start_s/end_s trimming is exercised too.

The numbers are made up. They only need to be roughly plausible so the code
runs end to end. Do not report results from this data.
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

LOCAL_IP = "10.0.0.23"
LOCAL_MAC = b"\x02\x00\x00\x00\x00\x23"
GW_MAC = b"\x02\x00\x00\x00\x00\x01"
LEAD_S = 5.0
SESSION_S = 180.0

SESSIONS = [
    # session_id,       case,           video, application, user
    ("idle_u1",         "idle",          "",   "idle",  1),
    ("videoA_u1",       "videoA_user1",  "A",  "video", 1),
    ("videoB_u1",       "videoB_user1",  "B",  "video", 1),
    ("web_u1",          "web_user1",     "",   "web",   1),
    ("videoA_u2",       "videoA_user2",  "A",  "video", 2),
]


def _frame(src, dst, proto, sport, dport, payload_len):
    payload = b"\x00" * max(payload_len, 0)
    if proto == "udp":
        l4 = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
        p = dpkt.ip.IP_PROTO_UDP
    else:
        l4 = dpkt.tcp.TCP(sport=sport, dport=dport, data=payload, flags=dpkt.tcp.TH_ACK)
        p = dpkt.ip.IP_PROTO_TCP
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=p, ttl=64, data=l4)
    outbound = src == LOCAL_IP
    eth = dpkt.ethernet.Ethernet(
        src=LOCAL_MAC if outbound else GW_MAC,
        dst=GW_MAC if outbound else LOCAL_MAC,
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


class Trace:
    def __init__(self):
        self.pkts: list[tuple[float, bytes]] = []

    def down(self, t, remote, size, proto="tcp", port=443):
        self.pkts.append((t, _frame(remote, LOCAL_IP, proto, port, 50000, size)))

    def up(self, t, remote, size, proto="tcp", port=443):
        self.pkts.append((t, _frame(LOCAL_IP, remote, proto, 50000, port, size)))


def _remote(rng, pool):
    return f"203.0.{pool}.{rng.integers(2, 250)}"


def background(tr, rng, t0, t1):
    """OS chatter present in every capture: DNS, NTP, telemetry."""
    servers = [_remote(rng, 113) for _ in range(3)]
    for t in np.sort(rng.uniform(t0, t1, int((t1 - t0) * 1.5))):
        s = servers[rng.integers(len(servers))]
        tr.up(t, s, int(rng.integers(60, 200)), "udp", 53)
        tr.down(t + 0.02, s, int(rng.integers(80, 300)), "udp", 53)


def video(tr, rng, t0, t1, chunk_kb, period_s, jitter, seeks=0):
    """Segment-based streaming: periodic bursts from one or two CDN hosts."""
    cdn = [_remote(rng, 7), _remote(rng, 7)]
    t = t0 + rng.uniform(0, 1)
    seek_times = set(np.round(rng.uniform(t0 + 20, t1 - 20, seeks)).astype(int))
    while t < t1:
        host = cdn[rng.integers(2)]
        size = chunk_kb * 1024 * rng.uniform(1 - jitter, 1 + jitter)
        if int(t) in seek_times or any(abs(int(t) - s) < period_s for s in seek_times):
            size *= 2.5  # rebuffer after a seek
        n = int(size // 1400)
        times = t + np.sort(rng.uniform(0, 0.8, n))
        for i, ti in enumerate(times):
            tr.down(ti, host, 1400, "udp")
            if i % 4 == 0:
                tr.up(ti + 0.005, host, 60, "udp")
        tr.up(t, host, int(rng.integers(300, 700)), "udp")  # segment request
        t += period_s * rng.uniform(0.85, 1.15)


def web(tr, rng, t0, t1):
    """Browsing: irregular page loads, many different servers per load."""
    t = t0 + rng.exponential(3)
    while t < t1:
        n_hosts = int(rng.integers(4, 18))
        for _ in range(n_hosts):
            host = _remote(rng, int(rng.integers(20, 90)))
            ts = t + rng.uniform(0, 2.5)
            tr.up(ts, host, int(rng.integers(400, 1400)))
            n = int(rng.lognormal(3.0, 1.0))
            for ti in ts + np.sort(rng.uniform(0.05, 1.0, n)):
                tr.down(ti, host, 1400)
                tr.up(ti + 0.003, host, 60)
        t += rng.exponential(9)


def build(session, rng):
    _sid, _case, vid, app, user = session
    tr = Trace()
    t_start, t_end = 1_700_000_000.0, 1_700_000_000.0 + 2 * LEAD_S + SESSION_S
    a, b = t_start + LEAD_S, t_start + LEAD_S + SESSION_S

    background(tr, rng, t_start, t_end)
    if app == "video":
        if vid == "A":
            kb, period = 600, 2.0
        else:
            kb, period = 1400, 4.0
        if user == 1:
            video(tr, rng, a, b, kb, period, jitter=0.15)
        else:  # user 2: more seeking, more head-motion upload
            video(tr, rng, a, b, kb, period, jitter=0.30, seeks=6)
            for t in np.arange(a, b, 0.05):
                tr.up(t, "203.0.7.9", int(rng.integers(90, 160)), "udp")
    elif app == "web":
        web(tr, rng, a, b)

    tr.pkts.sort(key=lambda x: x[0])
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for s in SESSIONS:
        tr = build(s, rng)
        path = out / f"{s[0]}.pcap"
        with open(path, "wb") as fh:
            w = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
            for t, buf in tr.pkts:
                w.writepkt(buf, ts=t)
        rows.append({
            "session_id": s[0], "file": path.name, "case": s[1],
            "video": s[2], "application": s[3], "user": s[4],
            # offsets from the first packet in the file
            "start_s": LEAD_S, "end_s": LEAD_S + SESSION_S,
            "local_ips": LOCAL_IP,
        })
        print(f"wrote {path} ({len(tr.pkts):,} packets)")
    pd.DataFrame(rows).to_csv(out / "sessions.csv", index=False)
    print(f"wrote {out / 'sessions.csv'}")


if __name__ == "__main__":
    main()
