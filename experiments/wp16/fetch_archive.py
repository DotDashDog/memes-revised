"""Fetch selected subreddit files from an Academic Torrents Reddit dump.

The dumps (Watchful1's "top 40k subreddits" series) are ~3.6 TiB across ~80k files;
we only ever want a couple of them, so this does a *selective* torrent download:
every file priority is set to 0 except the ones matching --want.

No torrent client is installed on this machine and installing one needs root, so
this drives libtorrent's Python bindings directly (`uv add libtorrent`).

Notes from the 2026-07-26 run (2025-12 cut, r/memes submissions + comments,
11.93 GB): 19.2 min wall clock, 4-6 peers, ~10 MB/s average after a slow first
minute. Piece length is 64 MiB, so selective download pulls some boundary data
belonging to neighbouring files -- that is the stray `.<infohash>.parts` file a
torrent client leaves behind, not corruption.

Download to the NVMe, not /mnt/d: torrent clients write pieces out of order, and
random writes on the portable drive's 5400 rpm platter over 9p are punishing.
Move to /mnt/d afterwards (ADR 0015 puts raw archives on the archive tier).

Known cuts (Academic Torrents infohashes):
    2023-12   56aa49f9653ba545f48df2e33679f014d2829c10   (dir: subreddits23)
    2024-12   1614740ac8c94505e4ecb9d88be8bed7b6afddd4
    2025-12   3e3f64dee22dc304cdd2546254ca1f8e8ae542b4   (dir: subreddits25)

Usage:
    curl -L -o cut.torrent https://academictorrents.com/download/<infohash>.torrent
    uv run python experiments/wp16/fetch_archive.py cut.torrent ~/reddit_dl \
        --want memes_submissions.zst memes_comments.zst
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import libtorrent as lt

DHT_ROUTERS = [
    ("router.bittorrent.com", 6881),
    ("dht.transmissionbt.com", 6881),
    ("router.utorrent.com", 6881),
]


def fetch(torrent: Path, save_path: Path, want: tuple[str, ...], poll: int = 30) -> None:
    ses = lt.session(
        {
            "listen_interfaces": "0.0.0.0:6881",
            "enable_dht": True,
            "enable_lsd": True,
            "enable_upnp": True,
            "enable_natpmp": True,
        }
    )
    for host, port in DHT_ROUTERS:
        ses.add_dht_router(host, port)

    info = lt.torrent_info(str(torrent))
    files = info.files()
    priorities = [0] * info.num_files()
    selected: list[tuple[str, int]] = []
    for i in range(info.num_files()):
        path = files.file_path(i)
        if path.endswith(want):
            priorities[i] = 7
            selected.append((path, files.file_size(i)))

    if not selected:
        raise SystemExit(f"no files in {torrent.name} matched {want}")

    total = sum(size for _, size in selected)
    print(f"selected {len(selected)} of {info.num_files():,} files, {total / 1e9:.2f} GB", flush=True)
    for path, size in selected:
        print(f"   {size / 1e9:6.2f} GB  {path}", flush=True)

    handle = ses.add_torrent(
        {"ti": info, "save_path": str(save_path), "file_priorities": priorities}
    )

    started, last = time.time(), 0.0
    while True:
        status = handle.status()
        if status.total_wanted and status.total_wanted_done >= status.total_wanted:
            print(f"\nCOMPLETE in {(time.time() - started) / 60:.1f} min", flush=True)
            break
        now = time.time()
        if now - last >= poll:
            pct = 100.0 * status.total_wanted_done / status.total_wanted if status.total_wanted else 0.0
            print(
                f"[{(now - started) / 60:6.1f} min] {pct:5.1f}%  "
                f"{status.total_wanted_done / 1e9:6.2f}/{status.total_wanted / 1e9:.2f} GB  "
                f"down {status.download_rate / 1e6:6.2f} MB/s  "
                f"peers {status.num_peers:3d}  seeds {status.num_seeds:3d}  {status.state}",
                flush=True,
            )
            last = now
        time.sleep(2)

    ses.remove_torrent(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("torrent", type=Path, help="path to the .torrent file")
    parser.add_argument("save_path", type=Path, help="download directory (use the NVMe)")
    parser.add_argument("--want", nargs="+", required=True, help="filename suffixes to download")
    parser.add_argument("--poll", type=int, default=30, help="seconds between progress lines")
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)
    fetch(args.torrent, args.save_path, tuple(args.want), args.poll)


if __name__ == "__main__":
    main()
