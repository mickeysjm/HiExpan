# HiExpan
The source code used for automatic taxonomy construction method HiExpan, published in KDD 2018.

## Requirments

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Also, a C++ compiler supporting C++11 is needed. 

If you want to use our pre-processing code, including corpus preprocessing and feature extraction pipeline, you need to install [SpaCy](https://spacy.io/usage/) and [gensim](https://radimrehurek.com/gensim/install.html). See detailed information in **/src/corpusProcessing/README.md** and **/src/featureExtraction/README.md**

## Step 1: Corpus pre-processing

To reuse our corpus pre-processing pipeline, you need to first create a data folder with dataset name `$DATA` at the project root, and then put your raw text corpus (each line represents a single document) under the `$DATA/source/` folder. 

```
data/$DATA
└── source
    └── corpus.txt
```

## Step 2: Feature extraction

You need to first transform the raw text corpus into a standard JSON format and use the code for feature extraction. The above pipeline will output two files, organized as follows:

```
data/$DATA
└── intermediate
	└── sentences.json
	└── entity2id.txt	
```

Based on these two files, feature extraction pipeline will output all the needed feature files for HiExpan model.

```
data/$DATA
└── intermediate
	└── eidSkipgramCounts.txt
	└── eidSkipgram2TFIDFStrength.txt
	└── eidTypeCounts.txt
	└── eidType2TFIDFStrength.txt
	└── eid2embed.txt
	└── eidDocPairPPMI.txt
```



