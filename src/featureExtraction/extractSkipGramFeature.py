"""
__author__: Ellen Wu, Jiaming Shen
__description__: extract skipgram features for candidate entities
    Input: 1) the sentence.json
    Output: 1) eidSkipgramCounts.txt, and 2) eidSentPairCount.txt
"""
import sys
import json
import itertools
import mmap
from tqdm import tqdm


def get_num_lines(file_path):
  fp = open(file_path, "r+")
  buf = mmap.mmap(fp.fileno(), 0)
  lines = 0
  while buf.readline():
    lines += 1
  return lines


def getSkipgrams(tokens, start, end):
  cleaned_tokens = []
  for tok in tokens:
    if tok == "\t":
      cleaned_tokens.append("TAB")
    else:
      cleaned_tokens.append(tok)
  positions = [(-1, 1), (-2, 1), (-3, 1), (-1, 3), (-2, 2), (-1, 2)]
  skipgrams = []
  for pos in positions:
    sg = ' '.join(cleaned_tokens[start+pos[0]:start]) + ' __ ' + ' '.join(cleaned_tokens[end+1:end+1+pos[1]])

    skipgrams.append(sg)
  return skipgrams


def processSentence(sent):
  sentInfo = json.loads(sent)
  eidSkipgrams = {}
  eidPairs = []
  tokens = sentInfo['tokens']
  eids = set()
  for em in sentInfo['entityMentions']:
    eid = em['entityId']

    start = em['start']
    end = em['end']
    eids.add(eid)

    for skipgram in getSkipgrams(tokens, start, end):
      key = (eid, skipgram)
      if key in eidSkipgrams:
        eidSkipgrams[key] += 1
      else:
        eidSkipgrams[key] = 1

  for pair in itertools.combinations(eids, 2):
    eidPairs.append(frozenset(pair))
  return eidSkipgrams, eidPairs


def writeMapToFile(map, outFilename):
  with open(outFilename, 'w') as fout:
    for key in map:
      lkey = list(key)
      fout.write(str(lkey[0])+'\t'+str(lkey[1])+'\t'+str(map[key])+'\n')


def updateMapFromMap(fromMap, toMap):
  for key in fromMap:
    if key in toMap:
      toMap[key] += fromMap[key]
    else:
      toMap[key] = fromMap[key]
  return toMap


def updateMapFromList(fromList, toMap):
  for ele in fromList:
    if ele in toMap:
      toMap[ele] += 1
    else:
      toMap[ele] = 1
  return toMap


def extractFeatures(dataname):
  outputFolder = '../../data/'+dataname+'/intermediate/'
  infilename = '../../data/'+dataname+'/intermediate/sentences.json'

  eidSkipgramCounts = {}
  eidPairCounts = {}  # entity sentence-level co-occurrence features
  with open(infilename, 'r') as fin:
    for line in tqdm(fin, total=get_num_lines(infilename),
                     desc="Generating skipgram and sentence-level co-occurrence features"):
      eidSkipgrams, eidPairs = processSentence(line)
      updateMapFromMap(eidSkipgrams, eidSkipgramCounts)
      updateMapFromList(eidPairs, eidPairCounts)

  writeMapToFile(eidSkipgramCounts, outputFolder+'eidSkipgramCounts.txt')
  writeMapToFile(eidPairCounts, outputFolder+'eidSentPairCount.txt')

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('Usage: extractSkipGramFeature.py -data')
    exit(1)
  corpusName = sys.argv[1]
  extractFeatures(corpusName)
