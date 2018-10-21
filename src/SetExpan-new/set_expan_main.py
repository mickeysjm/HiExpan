import argparse
import sys
sys.path.insert(0, '../HiExpan-new')
from dataLoader import loadEidToEntityMap, loadFeaturesAndEidMap, loadWeightByEidAndFeatureMap, \
    loadEntityEmbedding, loadEidDocPairPPMI
from set_expan_standalone import setExpan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='set_expan_main.py', description='Run SetExpan algorithm on input data')

    parser.add_argument('-data', required=True, help='CorpusName')
    parser.add_argument('-taxonPrefix', required=False, default="test", help='Output Taxonomy Prefix')
    # Model Parameters
    parser.add_argument('-max-iter-tree', type=int, default=5,
                        help='maximum iteration number of hierarchical tree expansion')
    parser.add_argument('-use-type', action='store_true', default=False,
                        help='use type feature or not, default is not use')
    parser.add_argument('-use-global-ref-edges', action='store_true', default=False,
                        help='use global reference edges or not, default is not use')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='debug mode or not, default is not debug')
    parser.add_argument('-num_initial_edge', type=int, default=1, help='number of each niece/nephew nodes for depth expansion')
    parser.add_argument('-num_initial_node', type=int, default=3,
                        help='number of each node\'s initial children for depth expansion')
    args = parser.parse_args()

    print("=== Start loading data ...... ===")
    folder = '../../data/' + args.data + '/intermediate/'
    eid2ename, ename2eid = loadEidToEntityMap(folder + 'entity2id.txt')
    eid2patterns, pattern2eids = loadFeaturesAndEidMap(folder + 'eidSkipgramCounts.txt')
    eidAndPattern2strength = loadWeightByEidAndFeatureMap(folder + 'eidSkipgram2TFIDFStrength.txt', idx=-1)
    eid2types, type2eids = loadFeaturesAndEidMap(folder + 'eidTypeCounts.txt')
    eidAndType2strength = loadWeightByEidAndFeatureMap(folder + 'eidType2TFIDFStrength.txt', idx=-1)
    eid2embed, embed_matrix, eid2rank, rank2eid, embed_matrix_array = loadEntityEmbedding(folder + 'eid2embed.txt')
    eidpair2PPMI = loadEidDocPairPPMI(folder + 'eidDocPairPPMI.txt')
    print("=== Finish loading data ...... ===")

    print("=== Start SetExpan ...... ===")
    seedEidsWithConfidence = [(8723, 1.0), (3362, 1.0), (10081, 1.0), (10320, 1.0), (7470, 1.0)]
    negativeSeedEids = set([])
    newOrderedChildrenEidsWithConfidence = setExpan(seedEidsWithConfidence, negativeSeedEids, eid2patterns,
                                                    pattern2eids, eidAndPattern2strength, eid2types, type2eids,
                                                    eidAndType2strength, eid2ename, eid2embed,
                                                    source_weights={"sg": 1.0, "tp": 1.0, "eb": 1.0},
                                                    max_expand_eids=5, use_embed=True,
                                                    use_type=True, FLAGS_VERBOSE=True, FLAGS_DEBUG=True)

    print("=== Finish SetExpan === \n{}\t{}\t{}".format("Entity ID", "Entity Name", "Confidence Score"))
    for ele in seedEidsWithConfidence:
        print("{}\t{}\t{}".format(ele[0], eid2ename[ele[0]], ele[1]))
    for ele in newOrderedChildrenEidsWithConfidence:
        print("{}\t{}\t{}".format(ele[0], eid2ename[ele[0]], ele[1]))