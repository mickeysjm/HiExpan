'''
__author__: Jiaming Shen
__description__: Run Stanford CoreNLP tool on AutoPhrase output and filter non-NP quality phrases.
    Assume the Stanford CoreNLP server is running on localhost:9002
    Input: 1) segmentation.txt
    Output: 1) a tmp sentences.json.raw (without entity information), 2) a quality NP phrases list
__latest_updates__: 08/25/2017
'''
import sys
import time
import json
import re
import multiprocessing
import os
from multiprocessing import Lock
from collections import deque
from pycorenlp import StanfordCoreNLP

class AutoPhraseOutput(object):
    def __init__(self, input_path, nlp):
        self.input_path = input_path
        self.nlp = nlp
        self.phrase_to_pos_sequence = {}  # key: lower case phrase, value: a dict of {"pos_sequence": count}
        self.pos_sequence_to_score = {}  # key: a pos sequence, value: a score in [0,0, 1.0]
        self.candidate_phrase = []

    def parse_one_doc(self, doc):
        """ Parse each document, update the phrase_to_pos_sequence, and convert the json format from Stanford
        CoreNLP to Ellen's sentences.json.raw format

        :param doc:
        :return:
        """
        ## replace non-ascii character
        doc = re.sub(r'[^\x00-\x7F]+', ' ', doc)
        ## add space before and after <phrase> tags
        doc = re.sub(r"<phrase>", " <phrase> ", doc)
        doc = re.sub(r"</phrase>", " </phrase> ", doc)
        ## add space before and after special characters
        doc = re.sub(r"([.,!:?()])", r" \1 ", doc)
        ## replace multiple continuous whitespace with a single one
        doc = re.sub(r"\s{2,}", " ", doc)

        res = self.nlp.annotate(doc, properties={
            "annotators": "tokenize,ssplit,pos",
            "outputFormat": "json"
        })
        output_sents = []
        for sent in res['sentences']:
            ## a new sentence
            output_token_list = []
            output_pos_list = []

            IN_PHRASE_FLAG = False
            q = deque()
            for token in sent['tokens']:
                word = token['word']
                pos = token['pos']
                if word == "<phrase>": # the start of a phrase
                    IN_PHRASE_FLAG = True
                    ## Mark the position of a phrase for postprecessing
                    output_token_list.append(word)
                    output_pos_list.append("START_PHRASE")
                elif word == "</phrase>": # the end of a phrase
                    ## obtain the information of current phrase
                    current_phrase_list = []
                    while (len(q) != 0):
                        current_phrase_list.append(q.popleft())
                    phrase = " ".join([ele[0] for ele in current_phrase_list]).lower() # convert to lower case
                    pos_sequence = " ".join([ele[1] for ele in current_phrase_list])

                    ## update phrase information
                    if phrase not in self.phrase_to_pos_sequence:
                        self.phrase_to_pos_sequence[phrase] = {}

                    if pos_sequence not in self.phrase_to_pos_sequence[phrase]:
                        self.phrase_to_pos_sequence[phrase][pos_sequence] = 1
                    else:
                        self.phrase_to_pos_sequence[phrase][pos_sequence] += 1

                    IN_PHRASE_FLAG = False

                    ## Mark the position of a phrase for postprecessing
                    output_token_list.append(word)
                    output_pos_list.append("END_PHRASE")
                else:
                    if IN_PHRASE_FLAG: # in the middle of a phrase, push the (word, pos) tuple
                        q.append((word,pos))

                    ## put all the token information into the output fields
                    output_token_list.append(word)
                    output_pos_list.append(pos)

            ## Finish processing one sentence, add the result into output_sents
            output_sents.append({
                "tokens": output_token_list,
                "pos": output_pos_list
            })

        if (len(q) != 0):
            print("[ERROR]: mismatched </phrase> in document: %s" % doc)

        return output_sents

    def save_phrase_to_pos_sequence(self, output_path=""):
        with open(output_path, "w") as fout:
            for phrase in self.phrase_to_pos_sequence:
                fout.write(phrase)
                fout.write("\t")
                fout.write(str(self.phrase_to_pos_sequence[phrase]))
                fout.write("\n")

    def load_phrase_to_pos_sequence(self, input_path=""):
        with open(input_path, "r") as fin:
            for line in fin:
                line = line.strip()
                seg = line.split("\t")
                if len(seg) < 2:
                    continue
                phrase = seg[0]
                pos_sequence = eval(seg[1])
                self.phrase_to_pos_sequence[phrase] = pos_sequence

    def score_pos_sequene(self):
        for pos_sequence_list in self.phrase_to_pos_sequence.values():
            for pos_sequence in pos_sequence_list:
                if pos_sequence not in self.pos_sequence_to_score:
                    if "NN" not in pos_sequence:
                        self.pos_sequence_to_score[pos_sequence] = 0.0
                    else:
                        self.pos_sequence_to_score[pos_sequence] = 1.0

    def obtain_candidate_phrase(self, threshold = 0.8, min_sup = 5):
        print("Number of phrases before filtering = %s" % len(self.phrase_to_pos_sequence))
        for phrase in self.phrase_to_pos_sequence:
            phrase_score = 0
            freq = sum(self.phrase_to_pos_sequence[phrase].values())
            if freq < min_sup:
                continue
            for pos_sequence in self.phrase_to_pos_sequence[phrase].keys():
                pos_sequence_weight = float(self.phrase_to_pos_sequence[phrase][pos_sequence]) / freq
                pos_sequence_score = self.pos_sequence_to_score[pos_sequence]
                phrase_score += (pos_sequence_weight * pos_sequence_score)
            if phrase_score >= threshold:
                self.candidate_phrase.append(phrase)
        print("Number of phrases after filtering = %s" % len(self.candidate_phrase))

    def save_candidate_phrase(self, output_path=""):
        with open(output_path, "w") as fout:
            for phrase in self.candidate_phrase:
                fout.write(phrase+"\n")

def process_corpus(input_path, output_path, output_phrase_to_pos_sequence_path, autoPhraseOutput):
    start = time.time()
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        # num_workers = 20
        # pool = mp.Pool(processes=num_workers)

        cnt = 0
        for line in fin:
            sentId = 0
            line = line.strip()
            if (cnt % 1000 == 0 and cnt != 0):
                cur = time.time()
                print("Processed %s documents, using time %s (seconds)" % (cnt, (cur-start)))
                # break
            try:
                sents = autoPhraseOutput.parse_one_doc(line)
            except:
                continue

            for sent in sents:
                sent["articleId"] = cnt
                sent["sentId"] = sentId
                sentId += 1
                json.dump(sent, fout)
                fout.write("\n")

            cnt += 1


    autoPhraseOutput.save_phrase_to_pos_sequence(output_phrase_to_pos_sequence_path)
    end = time.time()
    print("Finish Stanford CoreNLP processing, using time %s (second) " % (end-start))


def process_corpus_mp(input_path, output_path, output_phrase_to_pos_sequence_path, autoPhraseOutput):
    start = time.time()
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        num_workers = 20
        pool = multiprocessing.Pool(processes=num_workers)

        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        results = [pool.apply_async(
            process_corpus_thread, 
            args=(line, cnt)) for cnt, line in enumerate(fin)]
        results = [p.get() for p in results]
        
        sentId = 0
        for cnt, sents in enumerate(results):
            for sent in sents:
                sent['articleId'] = cnt
                sent['sentId'] = sentId
                sentId += 1
                json.dump(sent, fout)
                fout.write("\n")

    autoPhraseOutput.save_phrase_to_pos_sequence(output_phrase_to_pos_sequence_path)
    end = time.time()
    print("Finish Stanford CoreNLP processing, using time %s (second) " % (end-start))

def process_corpus_thread(line, cnt):
    if cnt % 1000 == 0:
        print("Processed %s documents; current time: %s" % (cnt, time.time()))
    line = line.strip()
    try:
        sents = autoPhraseOutput.parse_one_doc(line)
    except:
        return

    return sents

    """
        lock.aquire()
        json.dump(sent, fout)
        fout.write("\n")
        lock.release()
    """


if __name__ == '__main__':
    corpusName = sys.argv[1] # e.g. cs14confs
    FLAGS_POS_TAGGING = sys.argv[2]
    input_path = "../../data/"+corpusName+"/intermediate/segmentation.txt"
    output_path = "../../data/"+corpusName+"/intermediate/sentences.json.raw.tmp"
    output_phrase_to_pos_sequence_path = "../../data/"+corpusName+"/intermediate/phrase_to_pos_sequence.txt"
    output_phrase_path = "../../data/"+corpusName+"/intermediate/np_phrases.txt"
    nlp = StanfordCoreNLP('http://localhost:9000')
    autoPhraseOutput = AutoPhraseOutput(input_path=input_path, nlp=nlp)

    if FLAGS_POS_TAGGING:
        process_corpus(input_path=input_path, output_path=output_path,
                       output_phrase_to_pos_sequence_path=output_phrase_to_pos_sequence_path,
                       autoPhraseOutput=autoPhraseOutput)

    autoPhraseOutput.load_phrase_to_pos_sequence(output_phrase_to_pos_sequence_path)
    autoPhraseOutput.score_pos_sequene()
    autoPhraseOutput.obtain_candidate_phrase()
    autoPhraseOutput.save_candidate_phrase(output_phrase_path)
