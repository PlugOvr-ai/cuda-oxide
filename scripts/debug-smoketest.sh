#!/usr/bin/env bash
#
# scripts/debug-smoketest.sh -- end-to-end cuda-gdb validation of device
# debug info (CUDA_OXIDE_DEBUG=full).
#
# Builds an example with full device debug info and drives cuda-gdb in batch
# mode to prove that source debugging actually works on a real GPU: a source
# breakpoint binds, the backtrace is correct, and `info args` / `info locals`
# show real values (scalars, pointers, and struct fields), not just metadata
# in the emitted IR.
#
# This complements scripts/smoketest.sh (which validates the compile pipeline)
# by validating debugger *consumption* of the DWARF we emit.
#
# Gating: requires both cuda-gdb and a working NVIDIA GPU. When either is
# missing the script prints a skip notice and exits 0, so CI without a GPU is
# unaffected.
#
# Usage:
#   scripts/debug-smoketest.sh            # default example (compiler_features)
#   scripts/debug-smoketest.sh vecadd     # a specific example
#   CUDA_OXIDE_TARGET=sm_90 scripts/debug-smoketest.sh   # pin the arch

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="${1:-compiler_features}"
EXAMPLE_DIR="$REPO_ROOT/crates/rustc-codegen-cuda/examples/$EXAMPLE"

CUDA_GDB="${CUDA_OXIDE_CUDA_GDB:-$(command -v cuda-gdb || echo /usr/local/cuda/bin/cuda-gdb)}"

skip() {
    echo "debug-smoketest: SKIP ($1)"
    exit 0
}

[ -x "$CUDA_GDB" ] || skip "cuda-gdb not found (set CUDA_OXIDE_CUDA_GDB)"
command -v nvidia-smi >/dev/null 2>&1 || skip "nvidia-smi not found"
nvidia-smi -L >/dev/null 2>&1 || skip "no usable NVIDIA GPU / driver"
[ -d "$EXAMPLE_DIR" ] || { echo "debug-smoketest: FAIL (no example '$EXAMPLE')"; exit 1; }

# Resolve the device arch: explicit override wins, else the local GPU's cc.
if [ -n "${CUDA_OXIDE_TARGET:-}" ]; then
    ARCH="$CUDA_OXIDE_TARGET"
else
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    [ -n "$CC" ] || skip "could not read compute capability"
    ARCH="sm_${CC}"
fi

echo "debug-smoketest: example=$EXAMPLE arch=$ARCH"

# Build with full device debug info.
( cd "$REPO_ROOT" && CUDA_OXIDE_DEBUG=full CUDA_OXIDE_TARGET="$ARCH" \
    cargo oxide build "$EXAMPLE" ) || { echo "debug-smoketest: FAIL (build)"; exit 1; }

BIN="$EXAMPLE_DIR/target/release/$EXAMPLE"
[ -x "$BIN" ] || { echo "debug-smoketest: FAIL (no binary at $BIN)"; exit 1; }

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Drive cuda-gdb: stop at a kernel, walk to the kernel frame, dump args/locals.
GDB_LOG="$(mktemp)"
trap 'rm -f "$GDB_LOG"' EXIT
# Run from the example dir: the host binary resolves its embedded device
# artifact relative to the working directory.
( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
    -ex 'set pagination off' \
    -ex 'set breakpoint pending on' \
    -ex "break ${BREAK_AT:-test_option}" \
    -ex 'run' \
    -ex 'frame 1' \
    -ex 'info args' \
    -ex 'info locals' \
    -ex 'backtrace' \
    -ex 'kill' \
    "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1

echo "----- cuda-gdb output (tail) -----"
tail -25 "$GDB_LOG"
echo "----------------------------------"

# Verdict: a device breakpoint must have bound and fired, and at least one
# concrete value (scalar, pointer, or struct field) must be visible.
fail=0
grep -qiE "CUDA thread hit .*Breakpoint" "$GDB_LOG" || { echo "debug-smoketest: FAIL (no device breakpoint hit)"; fail=1; }
grep -qE "= [0-9]|0x[0-9a-f]|\{.*:" "$GDB_LOG"        || { echo "debug-smoketest: FAIL (no inspectable args/locals)"; fail=1; }
grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$GDB_LOG" && { echo "debug-smoketest: FAIL (PTX did not load under cuda-gdb)"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "debug-smoketest: PASS (source debugging + info args/locals verified on $ARCH)"
    exit 0
fi
exit 1
