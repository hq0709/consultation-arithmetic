#!/usr/bin/env bash
# Fetch the figure fonts into this folder (they are not committed).
#   Comic Sans MS 2.10: Microsoft core fonts for the web (redistributable only as the original package,
#   so it is downloaded here and extracted locally; embedding in PDF is permitted, fsType 0).
#   Inter 4.0: SIL Open Font License 1.1.
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o comic32.exe "https://downloads.sourceforge.net/corefonts/comic32.exe"
bsdtar -xf comic32.exe Comic.TTF Comicbd.TTF
mv Comic.TTF ComicSansMS-Regular.ttf && mv Comicbd.TTF ComicSansMS-Bold.ttf && rm comic32.exe
curl -sSL -o inter.zip "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
for f in Regular Medium SemiBold Bold Italic MediumItalic; do
  unzip -o -j -q inter.zip "extras/ttf/Inter-$f.ttf"
done
unzip -o -j -q inter.zip LICENSE.txt && mv LICENSE.txt Inter-LICENSE.txt && rm inter.zip
ls -1 *.ttf
