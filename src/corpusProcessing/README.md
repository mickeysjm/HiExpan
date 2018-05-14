### owner: Jiaming Shen, Ellen Wu, Dongming Lei

This folder contains scripts to 1) process raw text corpus into a standard formatted file for feature extraction, and 2) to perform key term extraction. 

First, you need to have [SpaCy](https://spacy.io/usage/) installed:

```
$ pip3 install -U spacy
$ python3 -m spacy download en
```

Then, you need to provide a raw txt file: "../../data/{corpus_name}/source/corpus.txt". Each line in corpus.txt represents a document. 

Finally, you can run the pre-processing pipeline by typing the following command:

```
$ chmod +x ./corpusProcess_new.sh
$ ./corpusProcess_new.sh $corpus_name $thread_number
```

### Example usage

For example, if your corpus_name is called "DBLP" and you want preprocess the data using 8 threads, you can first
put the raw text corpus in "../../data/DBLP/source/corpus.txt". Then, you can type the following command:

```
$ ./corpusProcess_new.sh DBLP 8
```






