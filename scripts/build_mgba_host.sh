#!/usr/bin/env bash
# Build the official mGBA core with Lua; no GUI or system installation required.
set -euo pipefail
cd "$(dirname "$0")/.."
root_dir="$PWD"
source_dir="$root_dir/.cache/mgba-source"
build_dir="$root_dir/.cache/mgba-build"
revision=26b7884bc25a5933960f3cdcd98bac1ae14d42e2
lua_prefix="${LUA_PREFIX:-$(brew --prefix lua@5.4)}"
mkdir -p .cache
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 --branch 0.10.5 https://github.com/mgba-emu/mgba.git "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'Unexpected mGBA revision' >&2; exit 1; }
cmake -S "$source_dir" -B "$build_dir" \
  -DBUILD_QT=OFF -DBUILD_SDL=OFF -DBUILD_SHARED=ON -DBUILD_STATIC=OFF \
  -DBUILD_GL=OFF -DBUILD_GLES2=OFF -DBUILD_GLES3=OFF \
  -DUSE_LUA=ON -DLUA_INCLUDE_DIR="$lua_prefix/include/lua" \
  -DLUA_LIBRARY="$lua_prefix/lib/liblua.dylib" \
  -DLUA_VERSION_MAJOR=5 -DLUA_VERSION_MINOR=4 -DLUA_VERSION_STRING=5.4 \
  -DUSE_FFMPEG=OFF -DUSE_LIBZIP=OFF -DUSE_DISCORD_RPC=OFF \
  -DBUILD_PYTHON=OFF -DCMAKE_PREFIX_PATH="$(brew --prefix)" \
  > .cache/mgba-configure.log 2>&1
cmake --build "$build_dir" -j "${JOBS:-6}" > .cache/mgba-build.log 2>&1
clang -std=c11 -Wall -Wextra -Werror \
  -I"$source_dir/include" -I"$build_dir/include" mgba/runner.c \
  -L"$build_dir" -lmgba "-Wl,-rpath,$build_dir" -o .cache/mgba-runner
printf 'Built %s/.cache/mgba-runner\n' "$root_dir"
