#!/bin/bash
DATA=$1
path=$(pwd)

## If you use local Probase dump for entity linking, set LINKING_THREAD=-1
## otherwise, set LINKING_THREAD to a value larger than 1
LINKING_THREAD=$2

## If you use local Probase dump for entity linking, set KB_PATH=../../KB/data-concept-instance-relations.txt
## If you do Probase linking by remote API, set KB_PATH=remote_API
KB_PATH=$3

## The embedding method used, currently support word2vec
EMBEDDING_METHOD=word2vec

## Number of threads used to learn word2vec embedding
EMBED_LEARNING_THREAD=$4

green=`tput setaf 2`
reset=`tput sgr0`
echo ${green}==='Corpus Name:' $DATA===${reset}
echo ${green}==='Current Path:' $path===${reset}


if [ ! -d ../../data/$DATA/intermediate ]; then
	mkdir ../../data/$DATA/intermediate
fi

if [ ! -d ../../data/$DATA/results ]; then
	mkdir ../../data/$DATA/results
fi


echo ${green}==='Extract Skipgram Features'===${reset}
python3 extractSkipGramFeature.py $DATA
python3 transformFeatures.py $DATA Skipgram

echo ${green}==='Extract Type Features (using Probase KB)'===${reset}
python3 probase3.py $DATA $LINKING_THREAD $KB_PATH
python3 extractTypeFeature.py $DATA
python3 transformFeatures.py $DATA Type

echo ${green}==='Extract Document-level Co-occurrence Features'===${reset}
python3 extractEidDocPairFeature.py $DATA

echo ${green}==='Extract Embedding Features (using word2vec)'===${reset}
python3 learnEmbedFeature.py $DATA $EMBED_LEARNING_THREAD