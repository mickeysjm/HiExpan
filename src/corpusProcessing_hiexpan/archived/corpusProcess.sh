#!/bin/bash
DATA=$1
path=$(pwd)
echo $path
# USE_CLEAN_PHRASE_LIST should be set to 0 in most cases
USE_CLEAN_PHRASE_LIST=${USE_CLEAN_PHRASE_LIST:- 0}
LANGUAGE=${LANGUAGE:- EN}
MIN_SUP=${MIN_SUP:- 10}
THREAD=${THREAD:- 4}

### Following clean the raw input data from corpus.txt -> corpus.clean.txt
echo ${green}===Cleaning input corpus===${reset}
python3 parseCorpus.py $DATA

### Following are the parameters used in auto_phrase.sh
RAW_TRAIN=${RAW_TRAIN:- ../../../data/$DATA/source/corpus.clean.txt}
if [ $USE_CLEAN_PHRASE_LIST -eq 1 ]; then
  MIN_SUP=${MIN_SUP:- 1000000000000}
  QUALITY_WIKI_ENTITIES=${QUALITY_WIKI_ENTITIES:- ../../../data/$DATA/source/clean_phrase_list.txt}
else
  # echo "concat phrase list"
  # cat ../../data/$DATA/source/clean_phrase_list.txt ../tools/AutoPhrase/data/EN/wiki_quality.txt > ../tools/AutoPhrase/data/phrase_combine.tmp
  # QUALITY_WIKI_ENTITIES=data/phrase_combine.tmp
  QUALITY_WIKI_ENTITIES=data/EN/wiki_quality.txt
fi

### Following are the parameters used in phrasal_segmentation.sh
HIGHLIGHT_MULTI=${HIGHLIGHT_MULTI:- 0.5}
HIGHLIGHT_SINGLE=${HIGHLIGHT_SINGLE:- 0.9}

green=`tput setaf 2`
reset=`tput sgr0`

echo ${green}==='Corpus Name:' $DATA===${reset}
echo ${green}==='Current Path:' $path===${reset}


#======= ENABLE THIS IF RUNNING WIKIDATA =========
# python3 getPhraseList.py $DATA

echo ${green}===Running AutoPhrase===${reset}
cd ../tools/AutoPhrase
make
echo ${green}==='RAW_TRAIN:' $RAW_TRAIN===${reset}
echo "auto_phrase.sh parameters:" $DATA $RAW_TRAIN $MIN_SUP $QUALITY_WIKI_ENTITIES $THREAD
./auto_phrase.sh $DATA $RAW_TRAIN $MIN_SUP $QUALITY_WIKI_ENTITIES $THREAD
echo "phrasal_segmentation.sh parameters:" $DATA $RAW_TRAIN $HIGHLIGHT_MULTI $HIGHLIGHT_SINGLE $THREAD
./phrasal_segmentation.sh $DATA $RAW_TRAIN $HIGHLIGHT_MULTI $HIGHLIGHT_SINGLE $THREAD

if [ ! -d ../../../data/$DATA/intermediate ]; then
  mkdir ../../../data/$DATA/intermediate
fi

cp models/$DATA/segmentation.txt ../../../data/$DATA/intermediate/segmentation.txt
cp models/$DATA/AutoPhrase_multi-words.txt ../../../data/$DATA/intermediate/AutoPhrase_multi-words.txt
cp models/$DATA/AutoPhrase_single-word.txt ../../../data/$DATA/intermediate/AutoPhrase_single-word.txt
cp models/$DATA/AutoPhrase.txt ../../../data/$DATA/intermediate/AutoPhrase.txt
cd $path

echo ${green}===Running spaCy Feature Extraction ===${reset}
python3 annotateNLPFeature.py $DATA 1 $THREAD