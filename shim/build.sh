#!/bin/sh
# Build libmeadow_torch against a libtorch install.
#
#   shim/build.sh            build into shim/build/
#   shim/build.sh test       build, then run the C smoke test
#   shim/build.sh install    build, then copy into $PREFIX/lib (default ~/.local/lib)
#
# LIBTORCH is a directory holding `include/` and `lib/`: an unpacked libtorch
# download, or the `torch` package directory of a Python install. Without it,
# $PREFIX/libtorch is used, and on aarch64 macOS downloaded there first if it
# is missing. On Linux, download the build for your CUDA version from
# pytorch.org and set LIBTORCH.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
prefix="${PREFIX:-$HOME/.local}"
libtorch_version=2.14.1

if [ -z "${LIBTORCH:-}" ]; then
  LIBTORCH="$prefix/libtorch"
  if [ ! -d "$LIBTORCH/include" ]; then
    if [ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]; then
      echo "no libtorch at $LIBTORCH: download one from pytorch.org and set LIBTORCH" >&2
      exit 1
    fi
    url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-$libtorch_version.zip"
    echo "downloading $url to $LIBTORCH" >&2
    mkdir -p "$prefix"
    curl -fL -o "$prefix/libtorch.zip" "$url"
    unzip -q -o "$prefix/libtorch.zip" -d "$prefix"
    rm "$prefix/libtorch.zip"
  fi
fi

case "$(uname -s)" in
  Darwin) ext=dylib; cxx=${CXX:-/usr/bin/clang++} ;;
  *)      ext=so;    cxx=${CXX:-c++} ;;
esac

# A libtorch built with CUDA registers its GPU backend only if libtorch_cuda is
# loaded, and nothing in the shim names a symbol from it, so the linker is told
# to keep it.
cuda=""
if [ -e "$LIBTORCH/lib/libtorch_cuda.so" ]; then
  cuda="-Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed"
fi

mkdir -p "$here/build"
out="$here/build/libmeadow_torch.$ext"

"$cxx" -std=c++20 -O2 -fPIC -shared -Wall -Wextra \
  -isystem "$LIBTORCH/include" -isystem "$LIBTORCH/include/torch/csrc/api/include" \
  "$here/meadow_torch.cpp" -o "$out" \
  -L"$LIBTORCH/lib" $cuda -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,"$LIBTORCH/lib"

echo "$out"

if [ "${1:-}" = "test" ]; then
  ${CC:-cc} -std=c11 -Wall -Wextra -I"$here" "$here/smoke_test.c" -o "$here/build/smoke_test" \
    -L"$here/build" -lmeadow_torch -lm -Wl,-rpath,"$here/build"
  (cd "$here/.." && "$here/build/smoke_test")
fi

if [ "${1:-}" = "install" ]; then
  dest="$prefix/lib"
  mkdir -p "$dest"
  cp "$out" "$dest/"
  echo "$dest/libmeadow_torch.$ext"
fi
