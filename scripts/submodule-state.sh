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
#
# Two classes of local modification are owned here:
#   submodule-<platform>.patch   exactly one active per machine (toolchain fixes)
#   submodule-tests.patch        applied on every platform (see COMMON_PATCH below)
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

# submodule-tests.patch is NOT a platform patch — it is applied on every platform.
#
# The three patches above answer a toolchain question, so exactly one of them is
# active per machine. This one fixes a defect in the reference library's OWN test
# suite that is identical on every OS, so it cannot belong to any of them: a fix
# parked in, say, the macOS patch would leave Linux and Windows running the broken
# test, which is how a per-platform patch file becomes where bugs go to hide.
#
# What it fixes: test/test_loss_layer.py::TTFSCrossEntropyLossTest::
# test_numerical_gradient_vs_error builds a TTFSCrossEntropyLoss with no w_in, so
# LIFLayer falls back to GaussianDistribution(mean 0, std 1) — and the test seeds
# nothing. Measured on this checkout: 50 fresh instances, 0 ever emitted a spike, so
# every first_spike_time was NaN. assert_allclose defaults to equal_nan=True, so the
# test can only pass when BOTH sides are NaN in the same places; it failed here
# because the analytical leg is legitimately 0.0 (nothing writes into the errors
# array when nothing spiked) while the numerical leg is NaN. A gradient test that
# compares NaN against 0.0, and passes only when both are NaN, measures nothing.
#
# The project's own Cython twin of this test
# (src/EventProp.Native.Python/test/test_loss_layer.py) already carries these five
# fixes — seed(42), explicit all-to-all w = 2.0, base times at 0.001, t_eps 1e-7,
# atol 1e-7 — so this patch mirrors them rather than inventing anything: the same
# defect, the same measured numbers (max |numerical| over the analytically-zero
# entries = 5.551e-10, worst relative error over the non-zero entries = 3.5e-9).
#
# It is still a modification to a read-only submodule, which is why it is owned here
# and reverted by `revert` alongside the platform patches.
COMMON_PATCH="$ROOT_DIR/submodule-tests.patch"

# Every patch this machine needs: the platform patch first, then the common one.
# Paths are emitted one per line; callers read them with `while IFS= read -r`.
#
# Both paths are emitted whether or not the file exists, because a patch that is
# configured but absent is a failure the caller must see: swallowing it here is how a
# submodule ends up half-patched and the build dies later inside pybind11/Eigen. The
# old code returned early with "no patch defined" and that was correct only for a
# platform that genuinely has no patch (PATCH_FILE empty), so emptiness is the sole
# skip condition; `apply_patch` dies on a named-but-missing file.
patch_list() {
  [ -n "${PATCH_FILE:-}" ] && printf '%s\n' "$PATCH_FILE"
  [ -n "${COMMON_PATCH:-}" ] && printf '%s\n' "$COMMON_PATCH"
  return 0
}

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
    note "no platform patch defined for OSTYPE='${OSTYPE:-unknown}' — skipping"
  fi
  local f
  while IFS= read -r f; do
    [ -f "$f" ] || die "patch not found: $f"
    apply_one_patch "$f"
  done < <(patch_list)
}

apply_one_patch() {
  local PATCH_FILE="$1"
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
  if [ -f "${COMMON_PATCH:-}" ]; then
    # Deliberately no "(expected ...)" excuse here: the common patch touches a file no
    # platform patch touches, so "blocked" can only mean upstream moved that file.
    printf '  %-24s %s\n' "submodule-tests.patch:" "$(patch_state "$COMMON_PATCH")  (all platforms)"
  fi
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
  local ok=true f
  while IFS= read -r f; do
    if ! is_applied "$f"; then
      echo "NOT BUILDABLE: $(basename "$f") is not applied" >&2; ok=false
    fi
  done < <(patch_list)
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
  # Every submodule-*.patch at the repo root, so a future patch file is drift-checked
  # the moment it exists instead of waiting for someone to remember this loop.
  for f in "$ROOT_DIR"/submodule-*.patch; do
    [ -f "$f" ] || continue
    p="$(basename "$f")"
    if (cd "$tmp" && git apply --check --exclude=src/pybind11 "$f") >/dev/null 2>&1; then
      printf '  %-24s %s\n' "$p:" "applies cleanly"
    else
      printf '  %-24s %s\n' "$p:" "DOES NOT APPLY — upstream drift, rebase required" >&2
      (cd "$tmp" && git apply --check --exclude=src/pybind11 "$f" 2>&1 | sed 's/^/      /') >&2 || true
      ok=false
    fi
  done
  $ok || exit 1
  echo "all patches are current"
}

cmd_revert() {
  echo "Restoring pristine EventProp.Python submodule..."
  local reverted=false f
  while IFS= read -r f; do
    if is_applied "$f"; then
      (cd "$SUBMODULE" && git apply -R "$f")
      note "reverted $(basename "$f")"
      reverted=true
    fi
  done < <(patch_list)
  $reverted || note "no patch to revert"
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
