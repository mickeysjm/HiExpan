"""
__author__: Ellen Wu, Jiaming Shen
__description__: Map entity surface to entity id and filter entities with too small occurrences.
    Input: 1) eidXXXCounts.txt (XXX is the feature name such as "Skipgram", "Type")
    Output: 1) eidXXX2TFIDFStrength.txt (XXX is the same feature name in input file)
"""
import sys
from math import log
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


def calculate_TFIDF_strength_new(inputFileName, outputFileName):
    eid_w_feature2count = defaultdict()  # mapping between (eid, feature) -> count
    feature2eidcount = defaultdict(int)  # number of distinct eids that match this feature
    feature2eidcountsum = defaultdict(int)  # total occurrence of eids matched this feature

    eid_set = set()
    with open(inputFileName, "r") as fin:
        for line in tqdm(fin, total=get_num_lines(inputFileName),
                         desc="Transform Features in {}".format(inputFileName)):
            seg = line.strip().split("\t")
            try:
                eid = seg[0]
                feature = seg[1]
                count = float(seg[2])
            except:
                print(seg)
                print(eid, feature, count)
                return

            eid_set.add(eid)
            eid_w_feature2count[(eid, feature)] = count
            feature2eidcount[feature] += 1
            feature2eidcountsum[feature] += count

    # Please refer to eq. (1) in http://mickeystroller.github.io/resources/ECMLPKDD2017.pdf
    print("[INFO] Start calculating TF-IDF strength")
    E = len(eid_set)  # vocabulary size
    with open(outputFileName, "w") as fout:
        for key in tqdm(eid_w_feature2count.keys(), desc="Process (eid, feature) pairs"):
            X_e_c = eid_w_feature2count[key]
            feature = key[1]
            f_e_c_count = log(1+X_e_c) * (log(E) - log(feature2eidcount[feature]))
            f_e_c_strength = log(1+X_e_c) * (log(E) - log(feature2eidcountsum[feature]))

            fout.write(key[0]+"\t"+key[1]+"\t"+str(f_e_c_count)+"\t"+str(f_e_c_strength)+"\n")


def main():
    corpusName = sys.argv[1]
    featureName = sys.argv[2]
    folder = '../../data/'+corpusName+'/intermediate/'
    inputFileName = folder+'eid'+featureName+'Counts.txt'
    outputFileName = folder+'eid'+featureName+'2TFIDFStrength.txt'
    calculate_TFIDF_strength_new(inputFileName, outputFileName)


if __name__ == '__main__':
    main()
