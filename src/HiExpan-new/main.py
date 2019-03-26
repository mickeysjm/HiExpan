import argparse
from collections import defaultdict
from util import getMostProbableNodeIdx
from dataLoader import loadEidToEntityMap, loadFeaturesAndEidMap, loadWeightByEidAndFeatureMap, \
    loadEntityEmbedding, loadEidDocPairPPMI
from seedLoader import load_seeds
from treeNode import TreeNode
from depthExpan import depth_expansion
from set_expan import setExpan
import math
import pickle
import os

# the maximum size of output taxonomy tree
level2max_children = {-1:15, 0:20, 1:40, 2:1e9, 3:1e9, 4:1e9, 5:1e9}
# the feature relative weights used for expanding a level x node's children
level2source_weights = {
    -1: {"sg":5.0, "tp":0.0, "eb":0.0},
    0: {"sg":5.0, "tp":0.0, "eb":0.0},
    1: {"sg":5.0, "tp":0.0, "eb":0.0},
    2: {"sg":5.0, "tp":0.0, "eb":0.0},
    3: {"sg":5.0, "tp":0.0, "eb":0.0},
    4: {"sg":5.0, "tp":0.0, "eb":0.0},
    5: {"sg":5.0, "tp":0.0, "eb":0.0},
}
# the maximum expanded entity number in each iteration under each node
level2max_expand_eids = {-1: 3, 0:5, 1:5, 2:5, 3:5, 4:5, 5:5}
# the global level-wise reference_edges between each two levels
level2reference_edges = defaultdict(list)


def obtainReferenceEdges(args, targetNode):
    if args.use_global_ref_edges:
        reference_edges = level2reference_edges[targetNode.level]
    else:
        reference_edges = []
        for sibling in targetNode.parent.children:
            cnt = 0
            for cousin in sibling.children:
                if cousin.isUserProvided and sibling.isUserProvided:
                    reference_edges.append((sibling.eid, cousin.eid))
                else:
                    reference_edges.append((sibling.eid, cousin.eid))
                    cnt += 1
                    if cnt >= args.num_initial_edge:
                        break
    return reference_edges


def isSynonym(args, eid1, eid2):
    eidpair = frozenset([eid1, eid2])
    for synset in args.synonyms_KB:
        if eidpair.issubset(synset):
            return True
    return False


def save_conflict_nodes(eidsWithConflicts, eid2nodes, conflict_nodes_file_path):
    with open(conflict_nodes_file_path, "w") as fout:
        fout.write("Number of conflict eids = %s\n" % len(eidsWithConflicts))
        fout.write("="*80+"\n")
        for eid in eidsWithConflicts:
            fout.write("Deal with conflict nodes with ename = %s, eid = %s\n" % (eid2ename[eid], eid))
            conflictNodes = eid2nodes[eid]
            for conflictNode in conflictNodes:
                fout.write("    "+str(conflictNode)+"\n")
            mostProbableNodeIdx = getMostProbableNodeIdx(conflictNodes, eid2patterns, eidAndPattern2strength)
            fout.write("!!!Most probable node:" + str(conflictNodes[mostProbableNodeIdx])+"\n")
            fout.write("="*80+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main.py', description='Run HiExpan algorithm on input data')

    parser.add_argument('-data', type=str, default="sample_dataset", help='CorpusName')
    parser.add_argument('-taxonPrefix', type=str, default="toy", help='Output Taxonomy Prefix')
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

    print("=== Start loading seed supervision ...... ===")
    userInput = load_seeds(args.data)
    if len(userInput) == 0:
        print("Terminated due to no user seed. Please specify seed taxonomy in seedLoader.py")
        exit(-1)

    stopLevel = max([ele[1] for ele in userInput]) + 1
    synonyms_KB = set([])
    args.synonyms_KB = synonyms_KB

    rootNode = None
    ename2treeNode = {}
    for i, node in enumerate(userInput):
        if i == 0:  # ROOT
            rootNode = TreeNode(parent=None, level=-1, eid=-1, ename="ROOT", isUserProvided=True, confidence_score=0.0,
                                max_children=level2max_children[-1])
            ename2treeNode["ROOT"] = rootNode
            for children in node[2]:
                newNode = TreeNode(parent=rootNode, level=0, eid=ename2eid[children], ename=children,
                                   isUserProvided=True, confidence_score=0.0, max_children=level2max_children[0])
                ename2treeNode[children] = newNode
                rootNode.addChildren([newNode])
        else:
            ename = node[0]
            eid = ename2eid[ename]  # assume user supervision is an entity mention in entity2id.txt
            level = node[1]
            childrens = node[2]
            if ename in ename2treeNode:  # existing node
                parent_treeNode = ename2treeNode[ename]
                for children in childrens:
                    newNode = TreeNode(parent=parent_treeNode, level=parent_treeNode.level + 1, eid=ename2eid[children],
                                       ename=children, isUserProvided=True, confidence_score=0.0,
                                       max_children=level2max_children[parent_treeNode.level + 1])
                    ename2treeNode[children] = newNode
                    parent_treeNode.addChildren([newNode])
                    level2reference_edges[parent_treeNode.level].append((parent_treeNode.eid, newNode.eid))
            else:  # not existing node
                print("[ERROR] disconnected tree node: %s" % node)
    print("=== Finish loading seed supervision ...... ===")

    rootNode.printSubtree(0)
    save_dir = "../../data/{}/results/{}".format(args.data, args.taxonPrefix)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    seed_taxonomy_file_path = "../../data/{}/results/{}/taxonomy_init.txt".format(args.data, args.taxonPrefix)
    rootNode.saveToFile(seed_taxonomy_file_path)

    print("=== Start HiExpan ...... ===")

    update = True
    iters = 0
    while update:
        eid2nodes = {}
        eidsWithConflicts = set()
        targetNodes = [rootNode] # targetNodes includes all parent nodes to expand
        while len(targetNodes) > 0:
            # expand childrens under current targetnode
            targetNode = targetNodes[0]
            targetNodes = targetNodes[1:]
            if targetNode.eid >= 0:
                eid = targetNode.eid

                # detect conflicts
                if eid in eid2nodes:
                    eid2nodes[eid].append(targetNode)
                    eidsWithConflicts.add(eid)
                else:
                    eid2nodes[eid] = [targetNode]

            # targetNode is already leaf node, stop expanding
            if targetNode.level >= stopLevel:
                continue

            # targetNode has enough children, just add children to be consider, stop expanding
            if len(targetNode.children) > targetNode.max_children:
                targetNodes += targetNode.children
                print("[INFO: Reach maximum children at node]:", targetNode)
                continue

            # Depth expansion: expand target node
            if len(targetNode.children) == 0:
                reference_edges = obtainReferenceEdges(args, targetNode)
                seedChildrenInfo = depth_expansion(reference_edges, targetNode.eid, eid2embed, embed_matrix_array,
                                                   rank2eid, eid2ename, embed_dim=100, topK=args.num_initial_node)
                if args.debug:
                    print("[Depth Expansion] Expand: {}".format(targetNode))
                    for ele in seedChildrenInfo:
                        print("\t\tObtain node with ename=%s, eid=%s" % (ele[1], ele[0]))

                seedOrderedChildren = []
                for seedChildren in seedChildrenInfo:
                    seedChildEid = seedChildren[0]
                    seedChildEname = seedChildren[1]
                    confidence_score = (targetNode.confidence_score + math.log(seedChildren[2]))
                    if seedChildEid != targetNode.eid:
                        seedOrderedChildren.append(TreeNode(parent=targetNode, level=targetNode.level+1, eid=seedChildEid,
                                                      ename=seedChildEname, isUserProvided=False,
                                                      confidence_score=confidence_score,
                                                      max_children=level2max_children[targetNode.level+1]))
                        level2reference_edges[targetNode.level].append((targetNode.eid, seedChildEid))
                targetNode.addChildren(seedOrderedChildren)

            # Wide expansion: obtain ordered new childrenEids
            seedEidsWithConfidence = [(child.eid, child.confidence_score) for child in targetNode.children]
            negativeSeedEids = targetNode.restrictions
            negativeSeedEids.add(targetNode.eid) # add parent eid as negative example into SetExpan
            if args.debug:
                print("[Width Expansion] Expand: {}, restrictions: {}".format(targetNode, negativeSeedEids))

            # at least grow one node
            max_expand_eids = max(len(negativeSeedEids)+1, level2max_expand_eids[targetNode.level])
            newOrderedChildrenEidsWithConfidence = setExpan(seedEidsWithConfidence, negativeSeedEids, eid2patterns,
                                                            pattern2eids, eidAndPattern2strength, eid2types, type2eids,
                                                            eidAndType2strength, eid2ename, eid2embed,
                                                            source_weights=level2source_weights[targetNode.level],
                                                            max_expand_eids=max_expand_eids, use_embed=True,
                                                            use_type=True)
            newOrderedChildren = []
            for ele in newOrderedChildrenEidsWithConfidence:
                newChildEid = ele[0]
                confidence_score = ele[1]
                confidence_score += targetNode.confidence_score
                synonym_FLAG = False  # Check synonmy
                for sibling in targetNode.children:
                    if isSynonym(args, newChildEid, sibling.eid):
                        sibling.addSynonym(newChildEid)
                        synonym_FLAG = True
                        if args.debug:
                            print("\t\t[Synonym] Find a pair of synonyms: <%s (%s), %s (%s)>" % (
                            sibling.ename, sibling.eid, eid2ename[newChildEid], newChildEid))
                        break
                if synonym_FLAG:  # bypass those synonym nodes
                    continue

                newChild = TreeNode(parent=targetNode, level=targetNode.level + 1, eid=newChildEid,
                                    ename=eid2ename[newChildEid], isUserProvided=False,
                                    confidence_score=confidence_score,
                                    max_children=level2max_children[targetNode.level + 1])

                if args.debug:
                    print("        Obtain node with ename=%s, eid=%s" % (eid2ename[newChildEid], newChildEid))
                newOrderedChildren.append(newChild)
            targetNode.addChildren(newOrderedChildren)

            # Add its children as in the queue
            targetNodes += targetNode.children

        # tree is expanded in this iter
        iters += 1
        taxonomy_file_path = "../../data/{}/results/{}/taxonomy_iter_{}_preprune.txt".format(args.data,
                                                                                             args.taxonPrefix, iters)

        rootNode.saveToFile(taxonomy_file_path)
        taxonomy_pickle_path = "../../data/{}/results/{}/taxonomy_iter_{}_preprune.pickle".format(args.data,
                                                                                             args.taxonPrefix, iters)
        with open(taxonomy_pickle_path, "wb") as fout:
            pickle.dump(rootNode, fout, protocol=pickle.HIGHEST_PROTOCOL)

        print("=== Starting Taxonomy Pruning at iteration %s ===" % iters)
        if args.debug:
            print("level2reference_edges:", level2reference_edges)
            rootNode.printSubtree(0)
            print("[INFO] Number of conflict eids at iteration %s = %s" % (iters, len(eidsWithConflicts)))
            conflict_nodes_file_path = "../../data/{}/results/{}/taxonomy_iter_{}_conflict_nodes.txt".format(args.data,
                                                                                                 args.taxonPrefix,
                                                                                                 iters)
            save_conflict_nodes(eidsWithConflicts, eid2nodes, conflict_nodes_file_path)

        # check conflicts
        nodesToDelete = []
        for eid in eidsWithConflicts:
            conflictNodes = eid2nodes[eid]
            mostProbableNodeIdx = getMostProbableNodeIdx(conflictNodes, eid2patterns, eidAndPattern2strength)
            for i in range(len(conflictNodes)):
              if i == mostProbableNodeIdx:
                continue
              nodesToDelete.append(conflictNodes[i])

        for node in nodesToDelete:
            node.parent.cutFromChild(node)
            node.delete()

        print("=== Taxonomy Tree at iteration %s ===" % iters)
        rootNode.printSubtree(0)
        taxonomy_file_path = "../../data/{}/results/{}/taxonomy_iter_{}_postprune.txt".format(args.data,
                                                                                              args.taxonPrefix, iters)
        rootNode.saveToFile(taxonomy_file_path)
        taxonomy_pickle_path = "../../data/{}/results/{}/taxonomy_iter_{}_postprune.pickle".format(args.data,
                                                                                                   args.taxonPrefix,
                                                                                                   iters)
        with open(taxonomy_pickle_path, "wb") as fout:
            pickle.dump(rootNode, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finish saving post-pruning Taxonomy Tree at iteration %s" % iters)

        if iters >= args.max_iter_tree:
            break

    print("Final Taxonomy Tree")
    rootNode.printSubtree(0)
    taxonomy_file_path = "../../data/{}/results/{}/taxonomy_final.txt".format(args.data, args.taxonPrefix)
    taxonomy_pickle_path = "../../data/{}/results/{}/taxonomy_final.pickle".format(args.data, args.taxonPrefix)
    with open(taxonomy_pickle_path, "wb") as fout:
        pickle.dump(rootNode, fout, protocol=pickle.HIGHEST_PROTOCOL)
    rootNode.saveToFile(taxonomy_file_path)
