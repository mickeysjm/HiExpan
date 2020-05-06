"""
__author__: Ellen Wu, Jiaming Shen
__description__: pre-processing the corpus before running AutoPhrase.
"""

import sys
import json
import re

corpusName = sys.argv[1]
inputFileName = '../../data/'+corpusName+'/source/corpus.txt'
outputFileName = '../../data/'+corpusName+'/source/corpus.clean.txt'

with open(inputFileName) as fin, open(outputFileName, 'w') as f_corpus:
    for doc in fin:
        doc = doc.strip()
        # remove non-ascii character
        doc = re.sub(r"[^\x00-\x7F]+", "", doc)

        # replace multiple continuous punctations
        # doc = re.sub(r"\!+", "!", doc)
        # doc = re.sub(r"\,+", ",", doc)
        # doc = re.sub(r"\?+", "?", doc)

        # add whitespace between/after some punctations
        # doc = re.sub(r"([.,!:?()])", r" \1 ", doc)

        # remove "\t" character
        doc = " ".join(doc.split("\t"))

        # replace multiple continuous whitespace with a single whitespace
        doc = re.sub(r"\s{2,}", " ", doc)
        if not doc:
            doc = "EMPTY_DOC_PLACEHOLDER"
        f_corpus.write(doc+'\n')
