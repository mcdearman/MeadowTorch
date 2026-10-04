#!/bin/sh
# Build libmeadow_torch against a libtorch install.
#
#   shim/build.sh            build into shim/build/
#   shim/build.sh test       build, then run the C smoke test
#   shim/build.sh install    build, then copy into $PREFIX/lib (default ~/.local/lib)
#
# LIBTORCH is a directory holding `include/` and `lib/`: an unpacked libtorch
# download, or the `torch` package directory of a Python install. By default it
# is the `torch` that $PYTHON (default python3) imports.
set -eu

here=$(cd "$(dirname "$0")" && pwd)

if [ -z "${LIBTORCH:-}" ]; then
  LIBTORCH=$("${PYTHON:-python3}" -c 'import os, torch; print(os.path.dirname(torch.__file__))') || {
    echo "no libtorch: set LIBTORCH, or PYTHON to a Python that has torch installed" >&2
    exit 1
  }
fi

case "$(uname -s)" in
  Darwin) ext=dylib; cxx=${CXX:-/usr/bin/clang++} ;;
  *)      ext=so;    cxx=${CXX:-c++} ;;
esac

mkdir -p "$here/build"
out="$here/build/libmeadow_torch.$ext"

"$cxx" -std=c++20 -O2 -fPIC -shared -Wall -Wextra \
  -isystem "$LIBTORCH/include" -isystem "$LIBTORCH/include/torch/csrc/api/include" \
  "$here/meadow_torch.cpp" -o "$out" \
  -L"$LIBTORCH/lib" -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,"$LIBTORCH/lib"

echo "$out"

if [ "${1:-}" = "test" ]; then
  ${CC:-cc} -std=c11 -Wall -Wextra -I"$here" "$here/smoke_test.c" -o "$here/build/smoke_test" \
    -L"$here/build" -lmeadow_torch -Wl,-rpath,"$here/build"
  "$here/build/smoke_test"
fi

if [ "${1:-}" = "install" ]; then
  dest="${PREFIX:-$HOME/.local}/lib"
  mkdir -p "$dest"
  cp "$out" "$dest/"
  echo "$dest/libmeadow_torch.$ext"
fi
