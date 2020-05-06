import sys
import json
from collections import defaultdict
import re

def loadMap(filename):
  map = {}
  with open(filename, 'r') as fin:
    for line in fin:
      if line:
        seg = line.strip('\r\n').split('\t')
        entity = re.sub("_", " ", seg[0])
        map[entity] = int(seg[-1])
  return map

def writeMapToFile(map, outFilename):
  with open(outFilename, 'w') as fout:
    for key in map:
      lkey = list(key)
      fout.write(str(lkey[0])+'\t'+str(lkey[1])+'\t'+str(map[key])+'\n')

if __name__ == "__main__":
  data = sys.argv[1]

  eidMapFilename = '../../data/'+data+'/intermediate/entity2id.txt'
  probaseLinkedFile = '../../data/'+data+'/intermediate/linked_results.txt'
  outFile = '../../data/' + data + '/intermediate/eidTypeCounts.txt'

  ent2eidMap = loadMap(eidMapFilename)

  eidType2count = defaultdict(float)
  with open(probaseLinkedFile) as fin:
    for line in fin:
      seg = line.strip('\r\n').split('\t')
      if len(seg) <= 1:
        continue
      entity = seg[0].lower()
      if entity not in ent2eidMap:
        print("[WARNING] No id entity: {}".format(entity))
        continue
      else:
        eid = ent2eidMap[entity]
      types = eval(seg[1])
      for tup in types:
        t = tup[0]
        strength = tup[1]
        eidType2count[(eid, t)] =  strength

  writeMapToFile(eidType2count, outFile)

