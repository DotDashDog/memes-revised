#!/usr/bin/env bash
# Real free disk space for this machine, as opposed to what `df /` says.
#
# Inside WSL, `df /` reports the virtual disk's nominal size (about 1 TB) no matter how
# much room the Windows drive holding that virtual disk actually has. On 2026-11-27 the
# virtual disk grew by 27 GB during a model download, filled C:, and WSL would not start.
# The numbers that matter are the free space on the Windows drive that holds the virtual
# disk, and on C: (the Windows system drive).
#
# Usage:
#   tools/host_disk.sh              print the real numbers
#   tools/host_disk.sh --need 30    exit 1 unless 30 GB can be written to the WSL disk
#                                   while leaving RESERVE_GB free on its host drive
#   tools/host_disk.sh --hook       Claude Code PreToolUse hook for Bash: blocks commands
#                                   when a host drive is nearly full (exit 2), except
#                                   commands that inspect or free space
set -u
RESERVE_GB=${RESERVE_GB:-25}      # keep at least this much free on the host drive
BLOCK_GB=${BLOCK_GB:-10}          # the hook blocks below this
CACHE="$HOME/.cache/wsl_host_drive"
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe

host_drive() {  # lowercase drive letter of the Windows drive holding this distro's virtual disk
  if [ -s "$CACHE" ] && [ -n "$(find "$CACHE" -mmin -720 2>/dev/null)" ]; then cat "$CACHE"; return; fi
  local d=""
  if [ -x "$PS" ]; then
    d=$(cd /mnt/c 2>/dev/null && "$PS" -NoProfile -Command \
      "(Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | ForEach-Object { Get-ItemProperty \$_.PSPath } | Where-Object DistributionName -eq '${WSL_DISTRO_NAME:-Ubuntu}').BasePath" \
      2>/dev/null | tr -d '\r' | sed -E 's/^\\\\\?\\//' | cut -c1 | tr 'A-Z' 'a-z')
  fi
  case "$d" in [a-z]) mkdir -p "$(dirname "$CACHE")"; echo "$d" > "$CACHE"; echo "$d";; *) echo "";; esac
}

free_gb() {  # free GB on a mounted Windows drive, empty if not mounted
  df -BG --output=avail "/mnt/$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

H=$(host_drive)
HF=""; [ -n "$H" ] && HF=$(free_gb "$H")
CF=$(free_gb c)

case "${1:-}" in
  --hook)
    cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)
    low=""
    [ -n "$HF" ] && [ "$HF" -lt "$BLOCK_GB" ] && low="drive ${H^^}: (holds the WSL disk) has ${HF} GB free"
    [ -n "$CF" ] && [ "$CF" -lt "$BLOCK_GB" ] && [ "$H" != "c" ] && low="${low:+$low; }drive C: (Windows) has ${CF} GB free"
    [ -z "$low" ] && exit 0
    case "$cmd" in  # let inspection and clean-up through
      df*|du*|ls*|rm\ *|find*|git\ status*|git\ gc*|uv\ cache*|*host_disk.sh*|nvidia-smi*|cat\ *|pgrep*|ps\ *|kill\ *|pkill\ *) exit 0;;
    esac
    echo "Blocked: $low, below the ${BLOCK_GB} GB floor. A full host drive stops WSL from starting. Free space or tell the user; only inspection and clean-up commands (df, du, ls, rm, find, uv cache, kill) run until then. See tools/host_disk.sh." >&2
    exit 2;;
  --need)
    need=${2:?GB required}
    [ -z "$HF" ] && { echo "cannot determine the host drive; check by hand" >&2; exit 1; }
    if [ $((HF - need)) -lt "$RESERVE_GB" ]; then
      echo "NOT ENOUGH ROOM: writing ${need} GB would leave $((HF - need)) GB on ${H^^}: (floor ${RESERVE_GB} GB)." >&2; exit 1
    fi
    echo "ok: ${HF} GB free on ${H^^}:, ${need} GB requested, $((HF - need)) GB would remain";;
  *)
    used=$(df -BG --output=used / | tail -1 | tr -dc '0-9')
    if [ -n "$H" ]; then echo "WSL disk lives on Windows drive : ${H^^}:"; else echo "WSL disk lives on Windows drive : unknown"; fi
    echo "Real free space on that drive   : ${HF:-unknown} GB   <- the limit for anything written inside WSL"
    echo "Real free space on C: (Windows) : ${CF:-unknown} GB"
    echo "Used inside WSL                 : ${used} GB"
    echo "Ignore 'df /': it reports the virtual disk's nominal size, not real space.";;
esac
