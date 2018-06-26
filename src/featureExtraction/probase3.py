"""
__author__: Dongming Lei, Jiaming Shen
__description__: Probase entity linking by either calling remote API or use local KB dump
"""
import sys,json,urllib.request,urllib.parse,os
import operator
import pickle
import time
import multiprocessing as mp
import math
import mmap
import ssl
from tqdm import tqdm
from collections import defaultdict
import re

FLAGS_DEBUG=False

def get_phrases(phrase_file):
    phrases = []
    with open(phrase_file, "r") as f:
        for line_id, line in tqdm(enumerate(f), total=get_num_lines(phrase_file), desc="Loading entity2id.txt"):
            line = line.rstrip()
            if line:
                entity= re.sub("_", " ", line.split("\t")[0])
                phrases.append(entity)
            else:
                print("[WARNING] wrong line id: {}".format(line_id))

    print('Finish obtaining {} phrases, ready for entity linking'.format(len(phrases)))
    return phrases


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


class ProbaseReference:
    """A class for Probase Linker using remote API"""
    _version = 1.0

    def __init__(self, corpusName, key = 'm3XGOzAbxDxNTjJQQ1gxlFQhXVkiBKl3'):
        self.corpusName = corpusName
        self.key = key
        self.api_url = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=%s&topK=10&api_key=' + key
        if os.path.isfile(corpusName + '-probase_local.p'):
            self.cache = self.load_from_file(corpusName + "-probase_local.p")
        else:
            self.cache = None

    def get_probase_fusion(self, phrase):
        if phrase in self.cache:
            return self.cache[phrase]
        else:
            res = self.get_probase_online(phrase)
            self.cache[phrase] = res
            self.save_to_file(res, "-probase_local.p")
            return res

    def get_probase_online(self, phrase):
        phrase = phrase.strip()
        if len(phrase) <=0:
            return None

        res = None
        for i in range(0,10):
            while True:
                try:
                    response = urllib.request.urlopen( self.api_url % urllib.parse.quote(phrase, safe=''), context=ssl._create_unverified_context() ).read()
                    res = json.loads(str(response, 'utf-8'))
                    res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
                except Exception as e:
                    time.sleep(10)
                    print('Retrying: %s' % phrase)
                    print(e)
                    continue
                break

        if FLAGS_DEBUG:
            if (res):
                print("[{}]Succeed in getting phrase: {}".format(
                    mp.current_process().name,
                    phrase,
                    )
                )
            elif (res == []):
                print("[{}]Unlinkable phrase:{}".format(
                    mp.current_process().name,
                    phrase,
                    )
                )
            else:
                print("[{}][ERROR] Failed to link:{}".format(
                    mp.current_process().name,
                    phrase,
                    )
                )
        return res

    def get_probase_batch(self, phrases, save = False):
        if phrases is None:
            return {}

        res = {}
        for e in tqdm(phrases):
            link_res = self.get_probase_online(e)
            res[e] = link_res

        if save:
            self.save_to_file(res, "-probase_local.p")

        return res

    def save_to_file(self, obj, file):
        pickle.dump(obj, open(file, "wb"))

    def load_from_file(self, file):
        return pickle.load(open(file, "rb"))

    def convert_to_txt_file(self, file):
        with open(file, "w") as fout:
            for ele in self.cache:
                fout.write("{}\t{}\n".format(ele, str(self.cache[ele])))

    def get_probase_parallel(
        self,
        phrases,
        num_workers = 1,
        save = False,
        save_file = None,
    ):

        num_workers+=1
        pool = mp.Pool(processes=num_workers)

        num_lines = len(phrases)
        batch_size = math.floor(num_lines/(num_workers-1))
        print("Linking by calling remote API using {} threads, {} phrases/threads".format(num_workers, batch_size))

        start_pos = [i*batch_size for i in range(0, num_workers)]
        phrases = list(phrases)
        results = [pool.apply_async(self.get_probase_batch, args=(phrases[start:start+batch_size], False)) for i, start in enumerate(start_pos)]
        results = [p.get() for p in results]

        res={}
        for r in results:
            res.update(r)

        if save:
            self.save_to_file(res, self.corpusName + "-probase_local.p")
            self.cache = res
            self.convert_to_txt_file(save_file)


class KnowledgeBase(object):
    """A class for Probase Linker using local KB dump, requires ~= 15GB memory for running """
    def __init__(self, corpusName, filepath=""):
        self.corpusName = corpusName
        self.entity2id = {}
        self.id2entity = {}
        self.num_entities = 0
        self.num_relations = 0

        self.eid2hypo = defaultdict(lambda: defaultdict(int))
        self.eid2hyper = defaultdict(lambda: defaultdict(int))
        self.eid2freq = defaultdict(int)

        if filepath:
            self._load_knowledgebase(filepath)

    def _get_num_lines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def _load_knowledgebase(self, filepath):
        """
        file format: hyper \t hypo \t cooccurrence
        """
        
        with open(filepath, "r") as fin:
            eid_count = 0
            for cnt, line in tqdm(enumerate(fin), total=self._get_num_lines(filepath),
                                  desc="Loading Probase (need ~= 15GB memory)"):
                line = line.strip()
                if not line:
                    continue

                segs = line.split("\t")
                self.num_relations += 1
                if segs[0] not in self.entity2id:
                    self.entity2id[segs[0]] = eid_count
                    self.id2entity[eid_count] = segs[0]
                    eid_count += 1
                    self.num_entities += 1
                if segs[1] not in self.entity2id:
                    self.entity2id[segs[1]] = eid_count
                    self.id2entity[eid_count] = segs[1]
                    eid_count += 1
                    self.num_entities += 1

                hyper = self.entity2id[segs[0]]
                hypo = self.entity2id[segs[1]]
                cooccurrence = float(segs[2])

                self.eid2hypo[hyper][hypo] = cooccurrence
                self.eid2freq[hyper] += cooccurrence
                self.eid2hyper[hypo][hyper] = cooccurrence
                self.eid2freq[hypo] += cooccurrence

    def linking(self, entity, topK=10):
        eid = self.entity2id.get(entity, -1)
        if eid == -1:
            return []
        else:
            hyper2BLC = {}
            for concept in self.eid2hyper[eid]:
                hyper2BLC[concept] = 1.0 * self.eid2hyper[eid][concept] ** 2 / (self.eid2freq[concept])

        top_concepts = sorted(hyper2BLC.items(), key=lambda x: -x[1])[:min(topK, len(hyper2BLC))]
        Z = sum([ele[1] for ele in top_concepts])
        return [(self.id2entity[ele[0]], ele[1] / Z) for ele in top_concepts]

    def save_to_file(self, obj, file):
        pickle.dump(obj, open(file, "wb"))

    def load_from_file(self, file):
        return pickle.load(open(file, "rb"))

    def convert_to_txt_file(self, file):
        with open(file, "w") as fout:
            for ele in self.cache:
                fout.write("{}\t{}\n".format(ele, str(self.cache[ele])))

    def get_probase(self, phrases, topK=10, save=True, save_file=None):
        print("Linking by using local Probase dump, {} phrases".format(len(phrases)))
        res = {}
        linkable = 0
        for phrase in tqdm(phrases):
            linked_types = self.linking(phrase, topK)
            res[phrase] = linked_types
            if linked_types:
                linkable += 1

        print("Number of linkable phrases = {} ({}).".format(linkable, 1.0 * linkable / len(phrases) ))
        if save:
            self.save_to_file(res, self.corpusName + "-probase_local.p")
            self.cache = res
            self.convert_to_txt_file(save_file)


if __name__ == "__main__":
    corpusName = sys.argv[1]
    num_workers = int(sys.argv[2])
    kb_path = sys.argv[3]
    phrase_file = "../../data/"+corpusName+"/intermediate/entity2id.txt"  # entity2id.txt
    save_file = "../../data/"+corpusName+"/intermediate/linked_results.txt"  # linked results
    phrases = get_phrases(phrase_file)

    start = time.time()
    if num_workers != -1:
        p = ProbaseReference(corpusName)
        p.get_probase_parallel(phrases, num_workers = num_workers, save=True, save_file=save_file)
    else:
        p = KnowledgeBase(corpusName, kb_path)
        p.get_probase(phrases, topK=10, save=True, save_file=save_file)

    end = time.time()
    print("Linking %s phrases using time %s (seconds)" % (len(phrases), end-start))




