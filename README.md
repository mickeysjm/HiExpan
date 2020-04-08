# HiExpan
The source code used for automatic taxonomy construction method [HiExpan](http://hanj.cs.illinois.edu/pdf/kdd18_jshen.pdf), published in KDD 2018. 

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

### Explanation of each intermediate files


1. **entity2id.txt**: each line has two columns (separated by a “\t” character) and represents one entity. The first column is entity surface name (with an underscore concatenating all words) and the second column is the entityID (which will be the unique identifier to retrieve each entity’s features).
2. **eidSkipgramCounts.txt**: each line has three columns (separated by a “\t” character). The first column is an entity id. The second column is a Skipgram feature associated with this entity. In the skipgram, the occurrence position of the entity is replaced with the placeholder “__”. Finally, the third column is the co-occurrence count between this entity id and the skipgram. For example, the line “0 \t reconstructed __ from \t 2” means “entity with id 0 appears twice in the context reconstructed __ from”. 
3. **eidSkipgram2TFIDFStrength.txt**: each line has four columns (separated by a “\t” character). The first and second columns are exactly the same as the eidSkipgramCounts.txt. The third and fourth columns are the association strength between entity and skipgram features. Larger values in third/fourth columns indicate stronger association between entity and skipgram features.
4. **eidTypeCounts.txt**: each line has three columns (separated by a “\t” character). The first column is an entity id. The second column is a type feature (in current version, the type is retrieved from Probase) associated with this entity. The third column is the probability that this entity has the corresponding type. For example, the line “2025 \t conditional simulation algorithm \t 0.251” means “the probability that entity with id 2025 is of type conditional simulation algorithm is 0.251”
5. **eidType2TFIDFStrength.txt**: each line has four columns (separated by a “\t” character). The first and second columns are exactly the same as the eidTypeCounts.txt. The third and fourth columns are normalized probability. Larger values in third/fourth columns indicate stronger association between entity and type features.
6. **eid2embed.txt**: each line is the embedding of one entity. This file is not human readable.
7. **eidDocPairPPMI.txt**: each line has three columns (separated by a “\t” character). The first and second columns are two entity ids. The third column is the Positive Pointwise Mutual Information (PPMI) behind these two entities. Larger values of PPMI indicate stronger association between these two entities.
8. **linked_results.txt**: each line has two columns (separated by a “\t” character). The first column is the entity surface name (no underscore) used as Probase linking input. The second column is the linking results. If an entity can not be linked, then the second column will simply be an empty list []. Otherwise, the second column will be a list of tuples and each tuple is (type name, linking probability). The linking probability indicates how likely an entity has the type. By analyzing this file, we can easily get how many entities are linkable to Probase. 


## Step 3: Taxonomy Construction

After obtaining all features for your corpus, you can provide seed taxonomy in **./HiExpan-new/seedLoader.py** and start running HiExpan model by the following commands:

```
$ cd ./HiExpan-new
$ python main -data $corpus_name -taxonPrefix $taxonPrefix
```
