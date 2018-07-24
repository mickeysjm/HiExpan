"""
__author__: Ellen Wu, Jiaming Shen
__description__: A bunch of data loading functions
"""
from collections import defaultdict
import mmap
from tqdm import tqdm
import numpy as np


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def loadEidToEntityMap(filename):
    eid2ename = {}
    ename2eid = {}
    with open(filename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(filename), desc="Loading: {}".format(filename)):
            seg = line.strip('\r\n').split('\t')
            eid2ename[int(seg[1])] = seg[0]
            ename2eid[seg[0]] = int(seg[1])
    return eid2ename, ename2eid


def loadFeaturesAndEidMap(filename):
    eid2feature = defaultdict(set)
    feature2eid = defaultdict(set)
    with open(filename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(filename), desc="Loading: {}".format(filename)):
            seg = line.strip('\r\n').split('\t')
            eid = int(seg[0])
            feature = seg[1]
            eid2feature[eid].add(feature)
            feature2eid[feature].add(eid)
    return eid2feature, feature2eid


def loadWeightByEidAndFeatureMap(filename, idx = -1):
    """Load the (eid, feature) -> strength

    :param filename:
    :param idx: The index column of weight, default is the last column
    :return:
    """
    weightByEidAndFeatureMap = {}
    with open(filename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(filename), desc="Loading: {}".format(filename)):
            seg = line.strip('\r\n').split('\t')
            eid = int(seg[0])
            feature = seg[1]
            weight = float(seg[idx])
            weightByEidAndFeatureMap[(eid, feature)] = weight
    return weightByEidAndFeatureMap


def loadEntityEmbedding(filename, dim=100):
    """ Load the entity embedding with word as context

    :param filename:
    :param dim: embedding dimension
    :return:
    """

    M = dim  # dimensionality of embedding
    eid2embed = {}
    embed_matrix = []
    cnt = 0
    eid2rank = {}
    rank2eid = {}
    with open(filename, "r") as fin:
        for line in tqdm(fin, total=get_num_lines(filename), desc="Loading: {}".format(filename)):
            # Assume no header
            seg = line.strip().split()
            eid2rank[int(seg[0])] = cnt
            rank2eid[cnt] = int(seg[0])

            embed = np.array([float(ele) for ele in seg[1:]])
            embed_matrix.append(embed)
            eid2embed[int(seg[0])] = embed.reshape(1, M)

            cnt += 1

        embed_matrix_array = np.array(embed_matrix)

    return eid2embed, embed_matrix, eid2rank, rank2eid, embed_matrix_array


def loadEidDocPairPPMI(filename):
    eidPair2PPMI = {}
    with open(filename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(filename), desc="Loading: {}".format(filename)):
            seg = line.strip('\r\n').split("\t")
            eid1 = int(seg[0])
            eid2 = int(seg[1])
            PPMI = float(seg[2])
            eidPair2PPMI[frozenset([eid1, eid2])] = PPMI
    return eidPair2PPMI


