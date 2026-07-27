# Environment & Package Management

This project uses **[uv](https://docs.astral.sh/uv/)** for Python interpreter, environment, and dependency management. All Python work goes through uv. Do not invoke system `python` or `pip` directly — that bypasses the locked environment and breaks reproducibility (see `01-scientific-rigor.md` §6).

## One-time setup (new contributor or agent)

1. **Install uv** (if you don't have it). Official installer:
  ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
   See [the docs](https://docs.astral.sh/uv/getting-started/installation/) for alternatives.
2. **Sync the environment** from the repo root:
  ```sh
   uv sync
  ```
   This will:
  - Install the Python version pinned in `.python-version` (currently 3.12) if not present.
  - Create `.venv/` (gitignored) and install everything from `uv.lock`.
3. **Verify**:
  ```sh
   uv run python -c "import sys; print(sys.version)"
  ```

You do **not** need to manually activate the venv. `uv run` handles that.

## Daily commands


| Task                                               | Command                                |
| -------------------------------------------------- | -------------------------------------- |
| Run a Python script                                | `uv run python path/to/script.py`      |
| Run a tool (e.g. pytest, jupyter)                  | `uv run pytest` / `uv run jupyter lab` |
| Add a runtime dependency                           | `uv add <pkg>`                         |
| Add a dev-only dependency                          | `uv add --dev <pkg>`                   |
| Remove a dependency                                | `uv remove <pkg>`                      |
| Sync env to lockfile                               | `uv sync`                              |
| Sync env **with GPU JAX** (Linux/WSL2 only)        | `uv sync --group gpu`                  |
| Update lockfile after manual `pyproject.toml` edit | `uv lock`                              |
| Bump Python version                                | edit `.python-version`, then `uv sync` |
| Quick REPL                                         | `uv run python`                        |


## Conventions

- **All dependencies live in `pyproject.toml`.** Runtime under `[project.dependencies]`, dev tools under `[dependency-groups].dev`. Ad-hoc `pip install` packages vanish on next `uv sync` and are never reproducible.
- `**uv.lock` is committed.** It is part of the project state. Regenerate it (`uv lock`) only when intentionally bumping versions, and commit the resulting diff.
- `**.python-version` pins the interpreter.** Bump deliberately; do not let it drift.
- `**.venv/` is never committed.** Already in `.gitignore`.
- `**uv run` for every Python invocation.** Scripts, notebooks, ad-hoc REPLs — all go through uv. This is a non-negotiable rule (see `/AGENTS.md`).
- **Pin within a sensible range.** Prefer `uv add 'numpy>=2.0,<3'` style for libraries with known major-version churn. For most things, `uv add numpy` is fine and `uv.lock` does the pinning.
- **Optional / experimental dependency groups.** For deps tied to a specific work package (e.g., a JAX vs. PyTorch choice not yet made), use `[dependency-groups]`. Install on demand with `uv sync --group <name>`. Document in the relevant work package.
- **Pre-existing code in the repo predates this setup.** If you run something from `memes_base/` or any legacy script and hit `ModuleNotFoundError`, add the dep via `uv add` and log what you added (and why) in the current work package. Do not assume legacy code's deps are still the right ones.

## GPU / ML libraries

Hardware: an NVIDIA 3090, which lives in a **separate Windows desktop**, not the laptop this repo is usually edited on. Plan accordingly — the laptop is CPU-only.

### The base `jax` dependency is CPU-only

`pyproject.toml` pins plain `jax`. That is deliberate: the CUDA wheels are Linux-only, so a bare `jax` keeps the lockfile resolvable on macOS. **GPU JAX comes from the `gpu` dependency group**, which is Linux-gated:

```sh
uv sync --group gpu   # no-op on macOS; installs jax[cuda12] on Linux/WSL2
```

Running it on a non-Linux machine is harmless (it resolves to nothing), so it is safe to put in a setup script.

### Platform support, and a silent-failure hazard

| Platform | JAX on NVIDIA GPU |
| --- | --- |
| Linux x86_64 | Yes |
| Windows WSL2 x86_64 | Experimental — but the only option on the Windows box |
| **Windows (native)** | **No.** JAX's docs: these installs "do not work with Windows, and may fail silently" |
| macOS | No (CPU only) |

The 3090 machine runs Windows, so **all GPU work must happen inside WSL2**, with the NVIDIA driver installed on the Windows host only (never inside WSL) and just the CUDA toolkit in the distro. Keep the repo on the WSL ext4 filesystem (`~/`), not `/mnt/c/...` — the 9p mount is drastically slower for the many-small-files image I/O this project does.

The hazard worth naming: a CPU-only JAX does not error. It produces correct-looking posteriors, just far slower. Per `method-discipline.md`, do not infer "the GPU is working" from a run completing. Check explicitly:

```sh
uv run python -c "import jax; print(jax.devices())"   # want CudaDevice, not CpuDevice
```

PyTorch, EasyOCR, and PaddleOCR do have native-Windows CUDA builds; it is specifically JAX (and therefore ADR 0013's primary numerical backend, WP-02, and WP-12) that forces WSL2.

### Verified desktop setup (2026-07-26)

Windows host `GAMING-PC` → WSL2 Ubuntu 26.04. 16 cores, 62 GB RAM, ~955 GB free.

- **No CUDA toolkit is installed in WSL, and none is needed.** The `gpu` group's pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-nvcc-cu12`, …) supply the whole CUDA runtime. Only the Windows-host NVIDIA driver is required (591.86 here, CUDA 13.1-capable). Do not `apt install cuda-toolkit`; it is a multi-GB no-op and invites version conflicts.
- **Nothing in the setup needs root.** `sudo` requires a password in this distro; uv installs to `~/.local/bin`.
- Repo lives at `~/memes-revised` on the WSL ext4 filesystem, not `/mnt/c`.

### Storage: two tiers, measured (2026-07-26)

Two devices, with very different characteristics. **Numbers below are measured on this box, not vendor claims.**

| Device | Path | Filesystem | Capacity | Free |
| --- | --- | --- | --- | --- |
| Samsung 970 EVO 1 TB (NVMe) | `/` → `~/memes-revised` | ext4 (native) | 1007 GB | 931 GB |
| Seagate Portable 4 TB (USB, 5400 rpm) | `/mnt/d` | NTFS over 9p/drvfs | 3726 GB | 3725 GB |

Benchmarked with the real WP-01 image cache (2296 JPEGs, 392 MB, 175 KB/image avg):

| Operation | Portable (`/mnt/d`) | NVMe (`/`) | Penalty |
| --- | --- | --- | --- |
| Sequential write, 2 GiB (`conv=fdatasync`) | 95 MB/s | 1.4 GB/s | 15× |
| Write 2296 small files | 15.5 s (148 files/s) | 0.33 s (~7000 files/s) | **47×** |
| Read 2296 small files | 9.1 s (252 files/s) | 0.02 s (page-cached) | — |
| Re-read, second pass | 9.1 s — **no improvement** | — | — |

Three conclusions that constrain pipeline design:

1. **95 MB/s sequential confirms a spinning platter.** The drive is a 5400 rpm 2.5" HDD behind a USB bridge (Windows reports `MediaType: Unspecified` because the bridge doesn't pass it through — infer it from the throughput, not the label).
2. **drvfs reads are not page-cached.** The second read pass was no faster than the first (9.12 s → 9.08 s), so a training loop over loose files on `/mnt/d` pays the full per-file cost *every epoch*. This is different from ext4, where the second pass is essentially free.
3. **Reformatting to ext4 would not fix the small-file case.** At 5400 rpm, truly random small reads are seek-bound at roughly 100 IOPS regardless of filesystem. The measured 252 files/s is already *better* than the platter's random ceiling only because the test files had just been written and were laid out contiguously — so treat 252 files/s as a **best case**, not a typical one. The bottleneck is the disk, not 9p; converting to ext4 would cost cross-platform readability and buy little.

Extrapolating to a 1M-image corpus (≈175 GB; the 3.7 TB drive holds ~21M images, so capacity is not the constraint):

- As **loose files** on `/mnt/d`: ~1.9 h of pure filesystem overhead to write, and ~66 min per read pass at the optimistic 252 files/s — realistically several hours once files are scattered, every epoch.
- As **packed shards** streamed sequentially at 95 MB/s: ~31 min per pass, predictable, and the 47× write penalty disappears.

So the portable drive is a good **archive and shard store** and a bad **working set**. See ADR 0015 for the resulting layout decision.

Setup, from a clean distro:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/DotDashDog/memes-revised.git ~/memes-revised
cd ~/memes-revised && uv sync --group gpu
```

Verification, and what a healthy result looks like:

```
$ uv run python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
gpu
[CudaDevice(id=0)]

8192^3 matmul: 0.030s  -> 37.2 TFLOP/s      # ~fp32 peak for a 3090; CPU would be <1
$ uv run pytest tests/                       # 4 passed
```

The throughput number is the point: `CudaDevice` alone only proves JAX *found* a GPU. A timed matmul proves it is *using* it. Prefer the measurement over the device list.

**Ignore this warning** — it is a benign WSL2 quirk (the driver reports a version string JAX's parser rejects) and does not affect execution:

```
cuda_executor.cc:1526] Could not get kernel mode driver version:
(INVALID_ARGUMENT: Version does not match the format X.Y.Z)
```

If a specific CUDA-version index is required, configure it in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

Document any index addition in this file's Log section below and explain why.

## Reproducibility checklist

Per `01-scientific-rigor.md` §6, every recorded experimental result should be reproducible from:

1. The git commit hash.
2. The committed `uv.lock`.
3. The recorded random seeds (in the experiment's config/log).
4. The recorded dataset identifier or checksum.

Do not bypass uv with `uv pip install` for anything you want kept. If you find yourself wanting to, stop and use `uv add`.

## Troubleshooting

- **"Why does `python script.py` fail but `uv run python script.py` work?"** You're hitting your system Python, which doesn't have the project deps. Always prefix with `uv run`.
- **"My VS Code linter doesn't see installed packages."** Point your editor at `.venv/bin/python` as the interpreter. uv creates that on `uv sync`.
- **"`uv sync` is slow on first run."** uv is downloading the Python toolchain and resolving the dep graph. Subsequent runs are fast.

## Log

- 2026-07-26 — Added `zstandard` (reading the archive dumps) and `libtorrent` (fetching new archive cuts). `libtorrent` is a **tooling** dependency, not a runtime one: no torrent client exists on this machine and installing one needs root, so the Python bindings are how new Academic Torrents cuts get pulled — see `experiments/wp16/fetch_archive.py`. Kept rather than removed because further cuts are planned; `uv remove libtorrent` is a one-liner if that changes. Raw archives now live at `/mnt/d/reddit_archive/` per ADR 0015; download to the NVMe first, since torrent clients write pieces out of order and random writes on the platter over 9p are punishing.
- 2026-07-26 — Mounted the 4 TB Seagate portable drive at `/mnt/d` (`mount -t drvfs D: /mnt/d -o metadata,uid=1000,gid=1000`; needs a password, so it is a user step, and WSL only auto-mounts drives attached before it starts) and **benchmarked both storage tiers with the real WP-01 image cache**. Headline: small-file writes are 47× slower on the portable drive, and drvfs reads get no page-cache benefit on a second pass, so loose-file image storage there costs full price every epoch. Sequential throughput (95 MB/s) identifies it as a 5400 rpm HDD; that also means reformatting to ext4 would not rescue random small reads, since the platter caps them near 100 IOPS. Table and extrapolations above. Feeds ADR 0015 (storage layout).
- 2026-07-26 — Stood up the WSL2 desktop environment and **verified GPU JAX end to end**: `jax.default_backend()` = `gpu`, `CudaDevice(id=0)`, 8192³ matmul at ~37 TFLOP/s (fp32 peak for a 3090), `tests/` 4 passed. Corrected the earlier setup guidance: the apt CUDA toolkit is *not* required, since the `gpu` group's pip wheels carry the CUDA runtime; only the Windows-host driver matters. No root needed anywhere. Claude Code 2.1.220 installed in the distro.
- 2026-07-26 — Added the Linux-gated `gpu` dependency group (`jax[cuda12]`, resolving CUDA 12.9 wheels) after finding the base `jax` pin was CPU-only on every platform, so the 3090 would have gone unused by ADR 0013's primary backend. Documented that native Windows has no JAX CUDA build (confirmed by JAX's docs *and* by the resolved lockfile marker, which excludes `win32` and `darwin`), so the 3090 desktop must run the project under WSL2. No ADR: this implements accepted ADR 0013 rather than changing the framework choice.
- 2026-05-19 — Added WP-02 runtime dependencies via uv: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; added dev dependency `pytest`. `uv.lock` is now part of the reproducible environment.
- 2026-05-19 — Bootstrapped uv: `pyproject.toml` (empty deps), `.python-version` = 3.12. `uv.lock` will be created on first `uv sync`.

