import gensim
from gensim.models import Word2Vec, KeyedVectors
import mmap
from tqdm import tqdm
import json
import re
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def minDuplicate(intervals):
    starts = []
    ends = []
    for i in intervals:
        starts.append(i[0])
        ends.append(i[1])

    starts.sort()
    ends.sort()
    s = e = 0
    numDuplicate = available = 0
    while s < len(starts):
        if starts[s] <= ends[e]:  # when an entity span starts, the previous entity span does not end
            if available == 0:   # if an available sentence sequence doesn't exit
                numDuplicate += 1
            else:
                available -= 1
            s += 1
        else:  # a new entity span starts after the previous entity span ends
            available += 1
            e += 1

    return numDuplicate


def processOneLine(sentInfo):
    """Return a (list of) sentence(s) with entity id replaced."""

    tokens = sentInfo["tokens"]
    if len(sentInfo["entityMentions"]) == 0:  # no entity mention in this sentence, simply return one sentence
        return [" ".join(tokens)]
    else:
        intervals = [[entityMention["start"], entityMention["end"]] for entityMention in sentInfo["entityMentions"]]
        num_duplicates = minDuplicate(intervals)
        if num_duplicates == 1:  # no overlapping entity mention
            for entityMention in sentInfo["entityMentions"]:
                eid = entityMention["entityId"]
                start = entityMention["start"]
                end = entityMention["end"]
                for i in range(start, end + 1):
                    tokens[i] = "ENTITY" + str(eid)
                    if i == start:
                        tokens[i] = "::" + tokens[i]
                    if i == end:
                        tokens[i] = tokens[i] + "::"

            sentence = " ".join(tokens)
            sentence = re.sub(r"::(ENTITY(\d+)\s*){1,}::", r"\1", sentence)
            return [sentence]
        else:
            res = []
            for mid in range(len(sentInfo["entityMentions"])):
                cur_tokens = tokens.copy()
                eid = sentInfo["entityMentions"][mid]["entityId"]
                start = sentInfo["entityMentions"][mid]["start"]
                end = sentInfo["entityMentions"][mid]["end"]
                for i in range(start, end + 1):
                    cur_tokens[i] = "ENTITY" + str(eid)
                sentence = " ".join(cur_tokens)
                sentence = re.sub(r"(ENTITY(\d+)\s*){1,}", r"\1", sentence)
                res.append(sentence)
            return res


def trim_rule(word, count, min_count):
    """Used in word2vec model to make sure entity tokens are preserved. """
    if re.match(r"^entity\d+$", word):  # keep entity token
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT


def extract_entity_embed_and_save(model, output_file):
    def match_rule(word):
        if re.match(r"^entity\d+$", word):
            return True
        else:
            return False

    model_size = model.vector_size
    vocab_size = len([word for word in model.wv.vocab if match_rule(word)])
    print("Saving embedding: model_size=%s,vocab_size=%s" % (model_size, vocab_size))
    with open(output_file, 'w') as f:
        for word in model.wv.vocab:
            if match_rule(word):
                vector_string = " ".join([str(ele) for ele in list(model.wv[word])])
                f.write("{} {}\n".format(word[6:], vector_string))


if __name__ == "__main__":
    corpusName = sys.argv[1]
    num_thread = int(sys.argv[2])
    inputFilePath = "../../data/"+corpusName+"/intermediate/sentences.json"
    saveFilePath = "../../data/"+corpusName+"/intermediate/eid2embed.txt"
    sentences = []
    with open(inputFilePath, "r") as fin:
        for line in tqdm(fin, total=get_num_lines(inputFilePath), desc="loading corpus for word2vec training"):
            sentInfo = json.loads(line.strip())
            sentence = processOneLine(sentInfo)
            for ele in sentence:
                sentences.append(ele.lower().split(" "))

    """
    Default configuration: learn 100-dim embedding using SkipGram model with Negative Sampling techniques
    Filter all tokens with frequency less than 5, except the entity token that satisfies the trim_rule.
    Pass through the corpus 5 iterations and set local context window length = 5, negative sample size = 12.
    Other related setting please refer to: https://radimrehurek.com/gensim/models/word2vec.html
    """
    model = Word2Vec(sentences, size=100, sg=1, hs=0, negative=12, window=5, min_count=5, workers=num_thread,
                     alpha=0.025, min_alpha=0.025 * 0.0001, sample=1e-3, iter=5, trim_rule=trim_rule)
    extract_entity_embed_and_save(model, saveFilePath)


