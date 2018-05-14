### owner: Ellen Wu, Jiaming Shen, Dongming Lei

This folder contains scripts to generate all feature files for HiExpan model. To run, simply change the corpus (data) name in main.sh and run ./dataProcess.sh

First, you need to have [gensim](https://radimrehurek.com/gensim/install.html) installed: 

```
pip3 install --upgrade gensim
```

Then, you need to provide 1) the corpus in standard JSON format file: "../../data/{corpus_name}/intermediate/sentences.json", and 2) an interested entity vocabulary: "../../data/{corpus_name}/intermediate/entity2id.txt". You may use the corpusProcessing pipeline to automatically generate these two files.

Finally, you can run the feature extraction pipeline using either local Probase dump or remote Probase API.

Run pipeline using remote Probase API:

```
$ chmod +x ./main.sh
$ ./main.sh $corpus_name $num_thread_for_linking remote_API $num_thread_for_word2vec
```

Run pipeline using local Probase dump:

```
$ chmod +x ./main.sh
$ ./main.sh $corpus_name -1 ../../KB/data-concept-instance-relations.txt $num_thread_for_word2vec
```
### Example usage

If your corpus_name is called "DBLP"; you want to use 30 threads to do entity linking based on remote Probase API, and you want to learn word2vec using 20 threads, you can type the command:

```
$ ./main.sh DBLP 30 remote_API 20
```

If your corpus_name is called "DBLP"; you want to do entity linking based on local Probase dump, and you want to learn word2vec using 20 threads, you can type the command:

```
$ ./main.sh DBLP -1 ../../KB/data-concept-instance-relations.txt 20
```




