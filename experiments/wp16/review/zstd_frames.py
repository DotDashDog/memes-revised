"""Walk the zstd container structure of a file by hand (frame headers and block headers,
per RFC 8878), without decompressing. Independent of any zstd library: answers "how many
frames does this file hold, and does the last one end exactly at end of file?" -- the
question that matters because python-zstandard's stream_reader stops at the end of the
first frame unless read_across_frames=True.

    uv run python experiments/wp16/review/zstd_frames.py FILE OUT_JSON
"""
import json
import os
import sys

path, out = sys.argv[1], sys.argv[2]
size = os.path.getsize(path)
frames = []
with open(path, "rb") as f:
    pos = 0
    while pos < size:
        f.seek(pos)
        magic = int.from_bytes(f.read(4), "little")
        if 0x184D2A50 <= magic <= 0x184D2A5F:
            n = int.from_bytes(f.read(4), "little")
            frames.append({"type": "skippable", "start": pos, "len": 8 + n})
            pos += 8 + n
            continue
        if magic != 0xFD2FB528:
            frames.append({"type": "GARBAGE", "start": pos, "magic": hex(magic)})
            break
        fhd = f.read(1)[0]
        fcs_flag, single, csum, did_flag = fhd >> 6, (fhd >> 5) & 1, (fhd >> 2) & 1, fhd & 3
        hdr = 4 + 1 + (0 if single else 1) + [0, 1, 2, 4][did_flag]
        fcs_len = [1 if single else 0, 2, 4, 8][fcs_flag]
        f.seek(pos + hdr)
        fcs = int.from_bytes(f.read(fcs_len), "little") if fcs_len else None
        if fcs is not None and fcs_len == 2:
            fcs += 256
        p = pos + hdr + fcs_len
        nblocks = 0
        while True:
            f.seek(p)
            b = f.read(3)
            if len(b) < 3:
                frames.append({"type": "TRUNCATED", "start": pos, "blocks": nblocks})
                pos = size + 1
                break
            h = int.from_bytes(b, "little")
            last, btype, bsize = h & 1, (h >> 1) & 3, h >> 3
            p += 3 + (1 if btype == 1 else bsize)
            nblocks += 1
            if last:
                break
        if pos > size:
            break
        p += 4 if csum else 0
        frames.append({"type": "zstd", "start": pos, "end": p, "blocks": nblocks,
                       "frame_content_size": fcs, "has_checksum": bool(csum)})
        pos = p
res = {"file": path, "bytes": size, "n_frames": len(frames),
       "ends_exactly_at_eof": bool(frames) and frames[-1].get("end") == size, "frames": frames[:20]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "frames"}), frames[:3])
