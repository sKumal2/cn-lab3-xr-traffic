"""Turn Wireshark captures into fixed 5-second feature windows.

Each window has exactly the four fields the lab asks for:
    download_bytes, upload_bytes, packet_count, remote_endpoints

Only packets to/from the laptop's own IP(s) are counted. Other people's
traffic that happens to be visible on the WiFi interface is ignored.
"""

from __future__ import annotations

import ipaddress
import socket
from collections import Counter
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

WINDOW_S = 5.0
FEATURES = ["download_bytes", "upload_bytes", "packet_count", "remote_endpoints"]


# --------------------------------------------------------------------------- #
# Reading packets
# --------------------------------------------------------------------------- #
def _open_reader(fh):
    magic = fh.read(4)
    fh.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


def _ip_layer(buf: bytes, linktype: int):
    """Return the dpkt IP/IP6 object for one frame, or None if not IP."""
    try:
        if linktype == dpkt.pcap.DLT_EN10MB:
            pkt = dpkt.ethernet.Ethernet(buf).data
        elif linktype in (dpkt.pcap.DLT_RAW, 101, 12):  # raw IP
            pkt = dpkt.ip.IP(buf) if buf[0] >> 4 == 4 else dpkt.ip6.IP6(buf)
        elif linktype == dpkt.pcap.DLT_LINUX_SLL:
            pkt = dpkt.sll.SLL(buf).data
        elif linktype in (dpkt.pcap.DLT_NULL, dpkt.pcap.DLT_LOOP):
            pkt = dpkt.loopback.Loopback(buf).data
        else:
            return None
    except (dpkt.dpkt.UnpackError, IndexError, ValueError):
        return None
    return pkt if isinstance(pkt, (dpkt.ip.IP, dpkt.ip6.IP6)) else None


def _addr(raw: bytes) -> str:
    fam = socket.AF_INET if len(raw) == 4 else socket.AF_INET6
    return socket.inet_ntop(fam, raw)


def read_packets(path: str | Path) -> pd.DataFrame:
    """Read a .pcap/.pcapng into a DataFrame: ts, length, src, dst."""
    rows = []
    with open(path, "rb") as fh:
        reader = _open_reader(fh)
        linktype = reader.datalink()
        for ts, buf in reader:
            ip = _ip_layer(buf, linktype)
            if ip is None:
                continue
            rows.append((float(ts), len(buf), _addr(ip.src), _addr(ip.dst)))
    return pd.DataFrame(rows, columns=["ts", "length", "src", "dst"])


def detect_local_ips(pkts: pd.DataFrame) -> set[str]:
    """Guess the laptop's address(es): the most frequent IPv4 and IPv6 address.

    Prefer passing local_ips explicitly in sessions.csv when you know them.
    """
    counts = Counter(pkts["src"]) + Counter(pkts["dst"])
    local = set()
    for version in (4, 6):
        candidates = [
            (n, a) for a, n in counts.items()
            if ipaddress.ip_address(a).version == version
            and not ipaddress.ip_address(a).is_multicast
        ]
        if candidates:
            local.add(max(candidates)[1])
    return local


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def windowize(
    pkts: pd.DataFrame,
    local_ips: set[str],
    start_s: float | None = None,
    end_s: float | None = None,
    window_s: float = WINDOW_S,
) -> pd.DataFrame:
    """Split one capture into non-overlapping windows.

    start_s / end_s are offsets (seconds) from the first packet in the file
    and mark the labelled part of the capture. Only *complete* windows are
    returned, and empty windows are kept as zero rows (important for "idle").
    """
    if pkts.empty:
        raise ValueError("capture contains no IP packets")

    t_first = pkts["ts"].min()
    t0 = t_first + (start_s or 0.0)
    t_end = t_first + end_s if end_s is not None else pkts["ts"].max()
    n_windows = int((t_end - t0) // window_s)
    if n_windows <= 0:
        raise ValueError("labelled segment is shorter than one window")

    p = pkts[(pkts["ts"] >= t0) & (pkts["ts"] < t0 + n_windows * window_s)].copy()
    up = p["src"].isin(local_ips)
    down = p["dst"].isin(local_ips)
    p = p[up | down]                      # drop traffic not involving us
    up, down = up[p.index], down[p.index]

    p["win"] = ((p["ts"] - t0) // window_s).astype(int)
    p["remote"] = np.where(up, p["dst"], p["src"])
    p["down_b"] = np.where(down & ~up, p["length"], 0)
    p["up_b"] = np.where(up & ~down, p["length"], 0)

    g = p.groupby("win")
    out = pd.DataFrame({
        "download_bytes": g["down_b"].sum(),
        "upload_bytes": g["up_b"].sum(),
        "packet_count": g.size(),
        "remote_endpoints": g["remote"].nunique(),
    })
    out = out.reindex(range(n_windows), fill_value=0)
    out.index.name = "window"
    return out.reset_index()


def capture_to_windows(path, local_ips=None, start_s=None, end_s=None):
    """Convenience: file -> windows DataFrame (+ the local IPs used)."""
    pkts = read_packets(path)
    ips = set(local_ips) if local_ips else detect_local_ips(pkts)
    return windowize(pkts, ips, start_s, end_s), ips
