#!/bin/sh
# Build the Chinese (ZH) preprint PDF from markdown + figures/.
# Requires: pandoc, xelatex, and the Noto Serif CJK SC font.
# No xeCJK/ctex needed — uses XeTeX core CJK line-breaking (see _pandoc-header-zh.tex).
cd "$(dirname "$0")"
pandoc preprint-draft-zh.md -o preprint-draft-zh.pdf \
  --pdf-engine=xelatex --toc \
  -V geometry:margin=2.2cm -V fontsize=11pt \
  -V mainfont="Noto Serif CJK SC" \
  -V monofont="DejaVu Sans Mono" \
  -V colorlinks=true \
  -H _pandoc-header-zh.tex
echo "wrote preprint-draft-zh.pdf"
