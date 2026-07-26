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

- 2026-07-26 — Added the Linux-gated `gpu` dependency group (`jax[cuda12]`, resolving CUDA 12.9 wheels) after finding the base `jax` pin was CPU-only on every platform, so the 3090 would have gone unused by ADR 0013's primary backend. Documented that native Windows has no JAX CUDA build (confirmed by JAX's docs *and* by the resolved lockfile marker, which excludes `win32` and `darwin`), so the 3090 desktop must run the project under WSL2. No ADR: this implements accepted ADR 0013 rather than changing the framework choice.
- 2026-05-19 — Added WP-02 runtime dependencies via uv: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; added dev dependency `pytest`. `uv.lock` is now part of the reproducible environment.
- 2026-05-19 — Bootstrapped uv: `pyproject.toml` (empty deps), `.python-version` = 3.12. `uv.lock` will be created on first `uv sync`.

