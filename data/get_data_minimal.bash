#!/usr/bin/env bash

fasttext='https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'

ZIPTOOL="unzip"

if [ "$OSTYPE" == "darwin"* ]; then
   # unzip can't handle large files on some MacOS versions
   ZIPTOOL="7za x"
fi

if [ ! -f ./fasttext/crawl-300d-2M.vec ] || [ ! -f ./fasttext/crawl-300d-2M.magnitude ]; then
    echo ${fasttext}
    mkdir fasttext
    curl -LO ${fasttext}
    ${ZIPTOOL} crawl-300d-2M.vec.zip -d fasttext/
    rm crawl-300d-2M.vec.zip
else
    echo "FastText vector file exists in the current directory"
fi

if [ ! -f ./fasttext/crawl-300d-2M.magnitude ]; then
    python -m pymagnitude.converter -i ./fasttext/crawl-300d-2M.vec -o ./fasttext/crawl-300d-2M.magnitude
    rm ./fasttext/crawl-300d-2M.vec
fi