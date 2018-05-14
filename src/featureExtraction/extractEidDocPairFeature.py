"""
__author__: Jiaming Shen
__description__: extract entity pair document level co-occurrence features
    Input: 1) the sentence.json
    Output: 1) eidDocPairCounts.txt, 2) eidDocPairPPMI.txt
"""
import sys
import json
import itertools
import math
from collections import defaultdict
import mmap
from tqdm import tqdm


def get_num_lines(file_path):
  fp = open(file_path, "r+")
  buf = mmap.mmap(fp.fileno(), 0)
  lines = 0
  while buf.readline():
    lines += 1
  return lines


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('Usage: python3 extractEidDocPairFeature.py -data')
    exit(1)
  corpusName = sys.argv[1]

  outfilename= '../../data/' + corpusName + '/intermediate/eidDocPairCount.txt'
  outfilename2 = '../../data/' + corpusName + '/intermediate/eidDocPairPPMI.txt'
  infilename = '../../data/' + corpusName + '/intermediate/sentences.json'

  articeID2eidlist = defaultdict(list)
  eid2freq = defaultdict(int)
  with open(infilename, 'r') as fin:
    for line in tqdm(fin, total=get_num_lines(infilename), desc="Generate document-level cooccurrence features (pass 1)"):
      sentInfo = json.loads(line)
      articleId = sentInfo['articleId']
      eidlist = [em['entityId'] for em in sentInfo['entityMentions']]
      articeID2eidlist[articleId].extend(eidlist)
      for eid in eidlist:
        eid2freq[eid] += 1


  eidPair2count = defaultdict(int)
  eidTotalCount = 0
  eidPairTotalCount = 0
  for articleId in tqdm(articeID2eidlist, desc="Generate document-level coocurrence features (pass 2)"):
    eidlist = articeID2eidlist[articleId]
    eidTotalCount += len(eidlist)
    for pair in itertools.combinations(eidlist,2):
      eidPairTotalCount += 1
      if pair[0] == pair[1]:
        continue
      eidPair2count[frozenset(pair)] += 1


  with open(outfilename, 'w') as fout:
    for ele in eidPair2count:
      count = eidPair2count[ele]
      ele = list(ele)
      fout.write(str(ele[0]) + "\t" + str(ele[1]) + "\t" + str(count) + "\n")

  with open(outfilename2, 'w') as fout:
    for ele in eidPair2count:
      p_x_y =  eidPair2count[ele] / eidPairTotalCount
      ele = list(ele)
      p_x = 1.0 * eid2freq[ele[0]] / eidTotalCount
      p_y = 1.0 * eid2freq[ele[1]] / eidTotalCount
      raw_pmi = math.log(p_x_y / (p_x * p_y) )
      if raw_pmi >= 0:
        ppmi = raw_pmi
      else:
        ppmi = 0.0
      fout.write(str(ele[0]) + "\t" + str(ele[1]) + "\t" + str(ppmi) + "\n")