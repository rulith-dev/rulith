#!/bin/sh
# Build the Chinese white paper PDF. Requires pandoc + xelatex + Noto Serif CJK SC.
cd "$(dirname "$0")"
pandoc whitepaper-self-driving-kernel-zh.md -o whitepaper-self-driving-kernel-zh.pdf \
  --pdf-engine=xelatex --toc \
  -V geometry:margin=2.2cm -V fontsize=11pt \
  -V mainfont="Noto Serif CJK SC" -V monofont="Noto Sans Mono CJK SC" -V colorlinks=true \
  -H _pandoc-header-zh.tex
echo "wrote whitepaper-self-driving-kernel-zh.pdf"
