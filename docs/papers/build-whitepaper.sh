#!/bin/sh
# Build the EN white paper PDF. Requires pandoc + xelatex + DejaVu Serif.
cd "$(dirname "$0")"
pandoc whitepaper-self-driving-kernel.md -o whitepaper-self-driving-kernel.pdf \
  --pdf-engine=xelatex --toc \
  -V geometry:margin=2.2cm -V fontsize=10pt -V mainfont="DejaVu Serif" -V colorlinks=true \
  -H _pandoc-header.tex
echo "wrote whitepaper-self-driving-kernel.pdf"
