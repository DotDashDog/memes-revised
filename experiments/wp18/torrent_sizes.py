"""WP-18: list sizes of the candidate subreddits' submissions files in a torrent (no download)."""
import sys
import libtorrent as lt

SUBS = ["memes", "dankmemes", "meirl", "comedyheaven", "whenthe", "196", "shitposting"]
info = lt.torrent_info(sys.argv[1])
print("infohash", info.info_hash(), "piece_len", info.piece_length(), "files", info.num_files())
fs = info.files()
want = {f"{s}_submissions.zst".lower(): s for s in SUBS}
tot = 0
for i in range(info.num_files()):
    p = fs.file_path(i)
    base = p.rsplit("/", 1)[-1]
    if base.lower() in want:
        sz = fs.file_size(i)
        print(f"{sz/1e9:8.3f} GB  {p}")
        if base.lower() != "memes_submissions.zst":
            tot += sz
print(f"total excluding memes: {tot/1e9:.3f} GB")
