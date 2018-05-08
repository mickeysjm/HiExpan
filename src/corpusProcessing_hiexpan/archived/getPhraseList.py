import sys

corpusName = sys.argv[1]
fname = '../../data/'+corpusName+'/source/entity2id_orig.txt' 
cleanPhraseListFile = '../../data/'+corpusName+'/source/clean_phrase_list.txt'
with open(fname) as fin, open(cleanPhraseListFile, 'w') as fout:
    for line in fin:
        segs = line.strip('\r\n').split('\t')
        fout.write(segs[0]+'\n')
