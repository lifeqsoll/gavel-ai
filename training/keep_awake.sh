#!/usr/bin/env bash
# Keep the session awake for overnight jobs: no sleep, no lock, no idle logout.
# Hyprland: pauses hypridle while active. systemd-inhibit blocks sleep/idle.
#
# Usage:
#   ./training/keep_awake.sh                 # until Ctrl+C
#   ./training/keep_awake.sh -- sleep 3600
#   ./training/keep_awake.sh -- ./training/overnight_hands_fix.sh
set -euo pipefail

WHY="${KEEP_AWAKE_WHY:-gavel overnight keep_awake}"
HYPRIDLE_WAS_ACTIVE=0
CLEANED=0

restore() {
  [[ "$CLEANED" -eq 1 ]] && return
  CLEANED=1
  if [[ "$HYPRIDLE_WAS_ACTIVE" -eq 1 ]]; then
    if command -v systemctl >/dev/null 2>&1; then
      systemctl --user start hypridle.service 2>/dev/null \
        || systemctl --user start hypridle 2>/dev/null \
        || true
    fi
    if ! pgrep -x hypridle >/dev/null 2>&1 && command -v hypridle >/dev/null 2>&1; then
      nohup hypridle >/dev/null 2>&1 &
    fi
    echo "keep_awake: hypridle restored" >&2
  fi
}
trap restore EXIT INT TERM

pause_hypridle() {
  if pgrep -x hypridle >/dev/null 2>&1; then
    HYPRIDLE_WAS_ACTIVE=1
    if systemctl --user is-active --quiet hypridle.service 2>/dev/null \
      || systemctl --user is-active --quiet hypridle 2>/dev/null; then
      systemctl --user stop hypridle.service 2>/dev/null \
        || systemctl --user stop hypridle 2>/dev/null \
        || true
    fi
    pkill -x hypridle 2>/dev/null || true
    echo "keep_awake: hypridle paused" >&2
  elif systemctl --user is-enabled --quiet hypridle.service 2>/dev/null; then
    HYPRIDLE_WAS_ACTIVE=1
    systemctl --user stop hypridle.service 2>/dev/null || true
    echo "keep_awake: hypridle.service stopped" >&2
  fi
}

# Nudge session so logind idle timers do not fire if something else watches idle.
nudge_loop() {
  while true; do
    if command -v hyprctl >/dev/null 2>&1; then
      hyprctl dispatch movecursor 0 0 >/dev/null 2>&1 || true
    fi
    if command -v busctl >/dev/null 2>&1; then
      busctl call org.freedesktop.ScreenSaver /org/freedesktop/ScreenSaver \
        org.freedesktop.ScreenSaver SimulateUserActivity >/dev/null 2>&1 || true
    fi
    sleep 60
  done
}

run_inhibited() {
  local -a cmd=("$@")
  if [[ "${#cmd[@]}" -eq 0 ]]; then
    echo "keep_awake: inhibiting idle/sleep until Ctrl+C ($WHY)" >&2
    cmd=(bash -c 'echo keep_awake: armed; while true; do sleep 3600; done')
  else
    echo "keep_awake: inhibiting idle/sleep while: ${cmd[*]}" >&2
  fi
  systemd-inhibit \
    --what=idle:sleep:handle-lid-switch \
    --who=gavel-keep-awake \
    --why="$WHY" \
    --mode=block \
    "${cmd[@]}"
}

pause_hypridle
nudge_loop &
NUDGE_PID=$!
trap 'kill "$NUDGE_PID" 2>/dev/null || true; restore' EXIT INT TERM

if [[ "${1:-}" == "--" ]]; then
  shift
  run_inhibited "$@"
else
  run_inhibited "$@"
fi
