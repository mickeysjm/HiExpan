### owner: Jiaming Shen, Ellen Wu, Dongming Lei

This folder contains scripts to generate input files for subsequent dataProcessing pipeline.

First, you need to have [SpaCy v2.x](https://spacy.io/usage/) installed:

```
$ pip3 install -U spacy
$ python3 -m spacy download en
```

Then, you need to provide a raw txt file: "../../data/{corpus_name}/source/corpus.txt".

Finally, you can run this pre-processing file by typing the following command:

```
$ chmod +x ./corpusProcess_new.sh
$ ./corpusProcess_new.sh $corpus_name $thread_number
```

For example, if your corpus_name is called "DBLP" and you want preprocess the data using 8 threads, you can first
put the raw text corpus in "../../data/DBLP/source/corpus.txt". Then, you can type the following command:

```
$ ./corpusProcess_new.sh DBLP 8
```






