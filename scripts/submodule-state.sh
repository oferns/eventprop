#!/usr/bin/env bash
# scripts/submodule-state.sh — manage the local modifications that the
# read-only EventProp.Python submodule needs in order to build on this machine.
#
#   ./scripts/submodule-state.sh status          # report state (read-only)
#   ./scripts/submodule-state.sh apply           # idempotent: patch + pybind11
#   ./scripts/submodule-state.sh verify          # exit non-zero if not buildable
#   ./scripts/submodule-state.sh check-patches   # do the patches survive upstream drift?
#   ./scripts/submodule-state.sh revert          # back to pristine submodule
#
# build.sh calls `apply`. Nothing here ever commits to the submodule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$ROOT_DIR/src/EventProp.Python"
PYBIND11_DIR="$SUBMODULE/src/pybind11"

# pybind11 is a NESTED submodule inside EventProp.Python, and the version the
# reference library needs is platform-specific:
#
#   recorded by the submodule : 54430436  pybind11 2.6.3.dev1 (2021)
#   required on macOS         : 6079989c  pybind11 3.1.0a0  (v3.0.2-48)
#   required on Linux         : a2e59f0e  pybind11 2.13.6   (encoded in the linux patch)
#
# The recorded 2.6.3.dev1 cannot build the reference library on a current
# toolchain — no NumPy 2 support, and it predates modern Clang/GCC and
# setuptools. On Linux the requirement is carried inside submodule-linux.patch
# (it moves the gitlink), so the patch is the source of truth there. On macOS no
# patch touches the gitlink, so the constant below is the only record of it.
#
# This is a required local modification, in the same class as the platform
# patches. A plain `git submodule update --init --recursive` silently downgrades
# it and breaks the reference build.
PYBIND11_RECORDED=54430436fee2afc4f8443691075a6208f9ea8eba
PYBIND11_REQUIRED_MACOS=6079989cf70555d213add249cfb36dad14519791

die()  { echo "ERROR: $*" >&2; exit 1; }
note() { echo "  → $*"; }

[ -e "$SUBMODULE/.git" ] || die "submodule not initialised: $SUBMODULE
  run: git submodule update --init src/EventProp.Python"

case "${OSTYPE:-}" in
  darwin*)              PATCH_FILE="$ROOT_DIR/submodule-macos.patch" ;;
  linux-gnu*)           PATCH_FILE="$ROOT_DIR/submodule-linux.patch" ;;
  msys*|cygwin*|mingw*) PATCH_FILE="$ROOT_DIR/submodule-windows.patch" ;;
  *)                    PATCH_FILE="" ;;
esac

# A git-format patch is "applied" when it reverse-applies cleanly, and
# "applicable" when it forward-applies cleanly. Both checks are non-destructive.
is_applied()    { (cd "$SUBMODULE" && git apply -R --check "$1") >/dev/null 2>&1; }
is_applicable() { (cd "$SUBMODULE" && git apply    --check "$1") >/dev/null 2>&1; }

patch_state() {
  if   is_applied     "$1"; then echo "applied"
  elif is_applicable  "$1"; then echo "applicable"
  else                         echo "blocked"
  fi
}

pybind11_head() { git -C "$PYBIND11_DIR" rev-parse HEAD 2>/dev/null || echo "unavailable"; }
describe()      { git -C "$1" describe --tags --always "$2" 2>/dev/null || echo "$2"; }

# The required pybind11 commit: whatever the active patch moves the gitlink to,
# else the per-platform constant, else the recorded pointer (no requirement).
pybind11_required() {
  local from_patch=""
  if [ -n "${PATCH_FILE:-}" ] && [ -f "${PATCH_FILE:-}" ]; then
    from_patch="$(sed -n 's/^+Subproject commit //p' "$PATCH_FILE" | head -1)"
  fi
  if [ -n "$from_patch" ]; then echo "$from_patch"; return; fi
  case "${OSTYPE:-}" in
    darwin*) echo "$PYBIND11_REQUIRED_MACOS" ;;
    *)       echo "$PYBIND11_RECORDED" ;;
  esac
}

apply_patch() {
  if [ -z "${PATCH_FILE:-}" ]; then
    note "no patch defined for OSTYPE='${OSTYPE:-unknown}' — skipping"
    return 0
  fi
  [ -f "$PATCH_FILE" ] || die "patch not found: $PATCH_FILE"

  local state; state="$(patch_state "$PATCH_FILE")"
  case "$state" in
    applied)
      note "$(basename "$PATCH_FILE") already applied" ;;
    applicable)
      (cd "$SUBMODULE" && git apply "$PATCH_FILE")
      note "applied $(basename "$PATCH_FILE")" ;;
    *)
      die "$(basename "$PATCH_FILE") does not apply cleanly to the submodule at $(git -C "$SUBMODULE" rev-parse --short HEAD).
  Run ./scripts/submodule-state.sh check-patches to see whether upstream has
  drifted. If it has, rebase the patch against the current submodule HEAD.
  Do NOT build on a partially patched submodule — the failure surfaces later as
  an opaque compile error inside pybind11/Eigen, far from the real cause." ;;
  esac
}

apply_pybind11() {
  [ -d "$PYBIND11_DIR" ] || die "nested pybind11 submodule not initialised:
  run: git -C src/EventProp.Python submodule update --init src/pybind11"

  local req head; req="$(pybind11_required)"; head="$(pybind11_head)"
  if [ "$head" = "$req" ]; then
    note "pybind11 already at required $(describe "$PYBIND11_DIR" "$req")"
    return 0
  fi
  if ! git -C "$PYBIND11_DIR" cat-file -e "$req" 2>/dev/null; then
    note "fetching pybind11 $req"
    (cd "$PYBIND11_DIR" && git fetch --tags --quiet origin) \
      || die "cannot fetch pybind11 $req from origin — fetch it manually"
  fi
  note "pybind11 $(describe "$PYBIND11_DIR" "$head") -> $(describe "$PYBIND11_DIR" "$req")"
  git -C "$PYBIND11_DIR" checkout -q "$req"
}

cmd_status() {
  echo "submodule: $(git -C "$SUBMODULE" rev-parse --short HEAD) $(describe "$SUBMODULE" HEAD)"
  echo "patches:"
  local active_state; active_state="$( [ -n "${PATCH_FILE:-}" ] && patch_state "$PATCH_FILE" || echo none )"
  local p f s
  for p in macos linux windows; do
    f="$ROOT_DIR/submodule-$p.patch"
    [ -f "$f" ] || continue
    s="$(patch_state "$f")"
    printf '  %-24s %s\n' "submodule-$p.patch:" "$s"
    if [ "$s" = "blocked" ] && [ "$active_state" = "applied" ] && [ "$f" != "${PATCH_FILE:-}" ]; then
      printf '  %-24s %s\n' "" "(expected: $(basename "$PATCH_FILE") occupies these hunks)"
    fi
  done
  echo "  active for this platform: $( [ -n "${PATCH_FILE:-}" ] && basename "$PATCH_FILE" || echo "none (OSTYPE=${OSTYPE:-unknown})" )"
  echo "pybind11 (nested submodule):"
  echo "  HEAD:      $(pybind11_head) $(describe "$PYBIND11_DIR" "$(pybind11_head)")"
  echo "  required:  $(pybind11_required) $(describe "$PYBIND11_DIR" "$(pybind11_required)")"
  echo "  recorded:  $PYBIND11_RECORDED (2.6.3.dev1 — does not build on macOS)"
  echo "reference binary:"
  local so; so=$(ls "$SUBMODULE"/eventprop/eventprop_cpp.*.so 2>/dev/null | head -1 || true)
  echo "  ${so:-not built}"
}

cmd_apply() {
  echo "Preparing EventProp.Python submodule for build..."
  apply_patch
  apply_pybind11
}

cmd_verify() {
  local ok=true
  if [ -n "${PATCH_FILE:-}" ] && ! is_applied "$PATCH_FILE"; then
    echo "NOT BUILDABLE: $(basename "$PATCH_FILE") is not applied" >&2; ok=false
  fi
  if [ "$(pybind11_head)" != "$(pybind11_required)" ]; then
    echo "NOT BUILDABLE: pybind11 at $(pybind11_head), required $(pybind11_required)" >&2; ok=false
  fi
  $ok || exit 1
  echo "submodule state OK (patch applied, pybind11 $(describe "$PYBIND11_DIR" "$(pybind11_required)"))"
}

# The only trustworthy drift test: do the patches apply to a PRISTINE submodule
# HEAD? On the live tree, foreign-platform patches read as "blocked" because the
# active patch owns the same hunks. A detached worktree at HEAD answers the real
# question without touching the working tree.
cmd_check_patches() {
  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/eventprop-patchcheck.XXXXXX")"
  git -C "$SUBMODULE" worktree add --detach -q "$tmp" HEAD
  # Expanded now, not at EXIT: `local tmp` is out of scope by then, and set -u
  # would make the trap itself die with "tmp: unbound variable".
  trap "git -C '$SUBMODULE' worktree remove --force '$tmp' >/dev/null 2>&1 || true" EXIT

  local ok=true p f
  local head; head="$(git -C "$SUBMODULE" rev-parse --short HEAD)"
  echo "checking patches against pristine submodule HEAD $head"
  echo "(gitlink hunks are excluded — nested submodules are not populated in the worktree)"
  for p in macos linux windows; do
    f="$ROOT_DIR/submodule-$p.patch"
    [ -f "$f" ] || continue
    if (cd "$tmp" && git apply --check --exclude=src/pybind11 "$f") >/dev/null 2>&1; then
      printf '  %-24s %s\n' "submodule-$p.patch:" "applies cleanly"
    else
      printf '  %-24s %s\n' "submodule-$p.patch:" "DOES NOT APPLY — upstream drift, rebase required" >&2
      (cd "$tmp" && git apply --check --exclude=src/pybind11 "$f" 2>&1 | sed 's/^/      /') >&2 || true
      ok=false
    fi
  done
  $ok || exit 1
  echo "all patches are current"
}

cmd_revert() {
  echo "Restoring pristine EventProp.Python submodule..."
  if [ -n "${PATCH_FILE:-}" ] && is_applied "$PATCH_FILE"; then
    (cd "$SUBMODULE" && git apply -R "$PATCH_FILE")
    note "reverted $(basename "$PATCH_FILE")"
  else
    note "no patch to revert"
  fi
  if [ -d "$PYBIND11_DIR" ]; then
    (cd "$SUBMODULE" && git submodule update --init src/pybind11) >/dev/null 2>&1 \
      && note "pybind11 restored to recorded $(describe "$PYBIND11_DIR" "$PYBIND11_RECORDED")" \
      || note "pybind11 left in place (restore manually: git -C src/EventProp.Python submodule update --init src/pybind11)"
  fi
  echo "Done. The reference library will NOT build in this state — run build.sh to re-prepare."
}

case "${1:-status}" in
  status)        cmd_status ;;
  apply)         cmd_apply ;;
  verify)        cmd_verify ;;
  check-patches) cmd_check_patches ;;
  revert)        cmd_revert ;;
  *) die "usage: $0 {status|apply|verify|check-patches|revert}" ;;
esac
