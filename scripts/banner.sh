#!/usr/bin/env bash
# Terminal-only release opener. Python generates the animated and static decks.
# NO_COLOR and NO_ANIM are opt-outs even when set to an empty value.
brand_style() {
  AE_INK= AE_QUIET= AE_TEAL= AE_AMBER= AE_CORAL= AE_RESET=
  AE_AMBER_SHADOW= AE_CORAL_SHADOW= AE_TEAL_SHADOW=
  if [[ -t 1 && ! ${NO_COLOR+x} && ${TERM:-} != dumb ]]; then
    AE_INK=$'\033[38;2;243;229;205m'
    AE_QUIET=$'\033[38;2;192;180;160m'
    AE_AMBER=$'\033[38;2;238;176;126m'
    AE_CORAL=$'\033[38;2;217;146;136m'
    AE_TEAL=$'\033[38;2;169;213;206m'
    AE_AMBER_SHADOW=$'\033[38;2;112;75;50m'
    AE_CORAL_SHADOW=$'\033[38;2;103;62;58m'
    AE_TEAL_SHADOW=$'\033[38;2;70;100;97m'
    AE_RESET=$'\033[0m'
  fi
}

_banner_plain() {
  _banner_short
}

_banner_short() {
  printf '%s\n' \
    'AEVONIX RESEARCH | GLM-5.3-Flash EXL3' \
    'TensorFold v0.6.6 + 86 patches | 4x RTX PRO 6000 | API :8020' \
    "TensorFold by Ash Hart · In collaboration with Mia's AI Lab"
}

banner() {
  [[ -t 1 ]] || return 0
  brand_style
  local script_dir=${BASH_SOURCE[0]%/*}
  local cols=${COLUMNS:-} rows=${LINES:-}

  if ! command -v python3 >/dev/null 2>&1; then
    if [[ ${NO_COLOR+x} || ${TERM:-} == dumb ]]; then
      _banner_plain
    else
      _banner_short
    fi
    return 0
  fi
  if [[ ${NO_COLOR+x} || ${TERM:-} == dumb ]]; then
    python3 "$script_dir/banner.py" --plain
    return
  fi

  # Bound the values before arithmetic; malformed or missing sizes use safe defaults.
  if [[ ! $cols =~ ^[1-9][0-9]{0,4}$ ]]; then
    cols=$(tput cols 2>/dev/null) || cols=80
  fi
  if [[ ! $rows =~ ^[1-9][0-9]{0,4}$ ]]; then
    rows=$(tput lines 2>/dev/null) || rows=24
  fi
  [[ $cols =~ ^[1-9][0-9]{0,4}$ ]] || cols=80
  [[ $rows =~ ^[1-9][0-9]{0,4}$ ]] || rows=24

  if (( cols < 86 )); then
    python3 "$script_dir/banner.py" --plain --columns "$cols"
  elif (( cols >= 112 && rows >= 32 )) && [[ ! ${NO_ANIM+x} && ( ${COLORTERM:-} == truecolor || ${COLORTERM:-} == 24bit ) ]]; then
    python3 "$script_dir/banner.py" --columns "$cols"
  else
    python3 "$script_dir/banner.py" --static --columns "$cols"
  fi
}
