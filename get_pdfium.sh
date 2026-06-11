#!/usr/bin/env bash
# Download a prebuilt pdfium dynamic library into ./lib/ for the current platform.
# Source: https://github.com/bblanchon/pdfium-binaries (BSD-licensed builds of Chrome's PDF engine)
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  target=mac-arm64 ;;
  Darwin-x86_64) target=mac-x64 ;;
  Linux-x86_64)  target=linux-x64 ;;
  Linux-aarch64) target=linux-arm64 ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

url="https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-${target}.tgz"
echo "Fetching ${url}" >&2

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/pdfium.tgz"
tar -xzf "$tmp/pdfium.tgz" -C "$tmp"

mkdir -p lib
cp "$tmp"/lib/libpdfium.* lib/
echo "Installed: $(ls lib/libpdfium.*)" >&2
