# HiExpan
The source code used for automatic taxonomy construction method HiExpan, published in KDD 2018.

## Requirments

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Also, a C++ compiler supporting C++11 is needed. 

If you want to use our data pre-processing code, you need to install SpaCy. See detailed information in **/src/corpusProcessing_hiexpan/README.md**

## Data pre-processing (Optional)

To reuse our data pre-processing pipeline, you need to first create a data folder with dataset name `$DATA` at the project root, and then put your raw text corpus (each line represents a single document) under the `$DATA/source/` folder. 

```
data/$DATA
└── source
    └── corpus.txt
```





