#!/usr/bin/env bash
# Apply the complete TensorFold v0.6.6 series in global patch-number order.
set -euo pipefail
export LC_ALL=C

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
pin=cb2ebf0540f42604e2759b2ddef497861e928248
fail() { printf 'apply-patches.sh: ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  echo 'usage: scripts/apply-patches.sh [CHECKOUT_OR_PACKAGE_DIR | --list | --help]'
  echo 'With no argument, clone TensorFold v0.6.6 into ./TensorFold and patch it.'
  echo '--list prints the ordered patch paths relative to this repository.'
}
list_only=0
target=
for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --list) [[ $list_only == 0 ]] || fail 'duplicate --list'; list_only=1 ;;
    -*) usage >&2; exit 2 ;;
    *) [[ -z "$target" ]] || fail 'provide only one target'; target=$arg ;;
  esac
done
[[ $list_only == 0 || -z "$target" ]] || fail '--list cannot be combined with a target'

# Sort by filename, across both origins. Directory order is not apply order.
mapfile -t patches < <(cd "$repo_dir" && printf '%s\n' patches/{miaai-lab,aevonix}/*.patch | sort -t / -k3,3)
[[ ${#patches[@]} == 86 ]] || fail 'expected exactly 86 release patches'
previous=
for p in "${patches[@]}"; do
  [[ -f "$repo_dir/$p" ]] || fail "missing patch series: $p"
  name=${p##*/}
  [[ "$name" =~ ^[0-9]{4}-.+\.patch$ ]] || fail "invalid patch name: $name"
  number=${name:0:4}
  [[ "$number" != "$previous" ]] || fail "duplicate patch number: $number"
  previous=$number
done
if [[ $list_only == 1 ]]; then printf '%s\n' "${patches[@]}"; exit 0; fi

if [[ -z "$target" ]]; then
  [[ ! -e TensorFold && ! -L TensorFold ]] || fail './TensorFold already exists; pass its path explicitly to use it'
  git clone --depth 1 --branch v0.6.6 https://github.com/ashhart/TensorFold ./TensorFold
  target=./TensorFold
  [[ "$(git -C "$target" rev-parse HEAD)" == "$pin" ]] || fail 'v0.6.6 does not match the pinned commit'
fi
[[ -d "$target" ]] || fail 'target must be an existing checkout or package directory'
target=$(cd -- "$target" && pwd -P)
if [[ -f "$target/src/tensorfold/__init__.py" ]]; then
  patch_dir=$target/src
elif [[ -f "$target/tensorfold/__init__.py" ]]; then
  patch_dir=$target
elif [[ "${target##*/}" == tensorfold && -f "$target/__init__.py" ]]; then
  patch_dir=${target%/*}
else
  fail 'cannot locate tensorfold/__init__.py in the target'
fi

check_artifacts() {
  local artifact
  artifact=$(find "$patch_dir/tensorfold" \( -name '*.rej' -o -name '*.orig' \) -print -quit)
  [[ -z "$artifact" ]] || fail 'target contains a .rej or .orig file; inspect and remove it before retrying'
}
check_artifacts
log=$(mktemp)
trap 'rm -f -- "$log"' EXIT
cd -- "$patch_dir"
for p in "${patches[@]}"; do
  # A dry run catches offsets before this patch can modify any files.
  for phase in check apply; do
    options=()
    [[ "$phase" != check ]] || options+=(--dry-run)
    if ! patch -p0 --forward --batch --fuzz=0 "${options[@]}" --input="$repo_dir/$p" >"$log" 2>&1; then
      cat "$log" >&2
      fail "$phase failed for ${p##*/}; the target may contain earlier patches from this run"
    fi
    if grep -Eiq '\b(fuzz|offset)\b' "$log"; then
      cat "$log" >&2
      fail "$phase reported fuzz or an offset for ${p##*/}"
    fi
    check_artifacts
  done
done
printf 'Applied %s patches to TensorFold v0.6.6 successfully (no fuzz, offsets, .rej or .orig).\n' "${#patches[@]}"
