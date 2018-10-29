#!/usr/bin/env bash

preprocess_exec="sed -f tokenizer.sed"

SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
MultiNLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'
fasttext='https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'
quora='http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
wikitext2='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
wikitext103='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
wikitext2-raw='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
wikitext3-raw='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'

ZIPTOOL="unzip"

if [ "$OSTYPE" == "darwin"* ]; then
   # unzip can't handle large files on some MacOS versions
   ZIPTOOL="7za x"
fi


# GloVe
 echo ${glovepath}
 mkdir GloVe
 curl -LO ${glovepath}
 ${ZIPTOOL} glove.840B.300d.zip -d GloVe/
 rm glove.840B.300d.zip

# fasttext
echo ${fasttext}
mkdir fasttext
curl -LO ${fasttext}
${ZIPTOOL} crawl-300d-2M.vec.zip -d fasttext/
rm crawl-300d-2M.vec.zip

### download SNLI
 mkdir SNLI
 curl -Lo SNLI/snli_1.0.zip ${SNLI}
 ${ZIPTOOL} SNLI/snli_1.0.zip -d SNLI
 rm SNLI/snli_1.0.zip
 rm -r SNLI/__MACOSX

 for split in train dev test
 do
     fpath=SNLI/${split}.snli.txt
     awk '{ if ( $1 != "-" ) { print $0; } }' SNLI/snli_1.0/snli_1.0_${split}.txt | cut -f 1,6,7 | sed '1d' > ${fpath}
     cut -f1 ${fpath} > SNLI/labels.${split}
     cut -f2 ${fpath} | ${preprocess_exec} > SNLI/s1.${split}
     cut -f3 ${fpath} | ${preprocess_exec} > SNLI/s2.${split}
     rm ${fpath}
 done
 rm -r SNLI/snli_1.0


# MultiNLI
# Test set not available yet : we define dev set as the "matched" set and the test set as the "mismatched"
 mkdir MultiNLI
 curl -Lo MultiNLI/multinli_0.9.zip ${MultiNLI}
 ${ZIPTOOL} MultiNLI/multinli_0.9.zip -d MultiNLI
 rm MultiNLI/multinli_0.9.zip
 rm -r MultiNLI/__MACOSX


 mv MultiNLI/multinli_0.9/multinli_0.9_train.txt MultiNLI/train.multinli.txt
 mv MultiNLI/multinli_0.9/multinli_0.9_dev_matched.txt MultiNLI/dev.matched.multinli.txt
 mv MultiNLI/multinli_0.9/multinli_0.9_dev_mismatched.txt MultiNLI/dev.mismatched.multinli.txt

 rm -r MultiNLI/multinli_0.9

 for split in train dev.matched dev.mismatched
 do
     fpath=MultiNLI/${split}.multinli.txt
     awk '{ if ( $1 != "-" ) { print $0; } }' ${fpath} | cut -f 1,6,7 | sed '1d' > ${fpath}.tok
     cut -f1 ${fpath}.tok > MultiNLI/labels.${split}
     cut -f2 ${fpath}.tok | ${preprocess_exec} > MultiNLI/s1.${split}
     cut -f3 ${fpath}.tok | ${preprocess_exec} > MultiNLI/s2.${split}
     rm ${fpath} ${fpath}.tok
 done

# quora
echo ${quora}
mkdir quora
curl -Lo quora/quora_duplicate_questions.tsv ${quora}

# wikitext-2
echo ${wikitext2}
mkdir wikitext2
curl -Lo wikitext2/wikitext-2-v1.zip ${wikitext2}
${ZIPTOOL} wikitext2/wikitext-2-v1.zip -d wikitext2
rm wikitext2/wikitext-2-v1.zip

echo ${wikitext103}
mkdir wikitext103
curl -Lo wikitext103/wikitext-103-v1.zip ${wikitext103}
${ZIPTOOL} wikitext103/wikitext-103-v1.zip -d wikitext103
rm wikitext103/wikitext-103-v1.zip

