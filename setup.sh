#!/bin/bash

set -e

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz
mkdir -p embeddings/
mv cc.en.300.bin embeddings/

wget https://downloads.dbpedia.org/repo/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2
bzip2 -dk alias.ttl.bz2
rm alias.ttl.bz2
