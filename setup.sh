#!/bin/bash

set -e

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz
mkdir -p embeddings/
mv cc.en.300.bin embeddings/

python3 generate_aliases.py
