"""
__author__: Ellen Wu (modified by Jiaming Shen)
__description__: A bunch of utility functions
"""


def hasCausalRelationship(pathx, pathy):
    if len(pathx) > len(pathy):
        return False
    else:
        for i in range(len(pathx)-1):
            if pathx[i] != pathy[i]:
                return False
        if pathx[len(pathx)-1] > pathy[len(pathx)-1]:
            return False
        else:
            return True


def getMostProbableNodeIdx(nodes, eid2patterns, eidAndPattern2strength):
    """
    :param nodes: a list of TreeNode objects
    :param eid2patterns:
    :param eidAndPattern2strength:
    :return:
    """
    # Sanity checking
    assert len(nodes) > 0
    eid = nodes[0].eid
    for i in range(1, len(nodes)):
        assert nodes[i].eid == eid

    # Rule 1: User is the big boss
    for i in range(len(nodes)):
        if nodes[i].isUserProvided:
            return i

    candidates = set(range(len(nodes)))
    treePaths = []
    for i in range(len(nodes)):
        node = nodes[i]
        path = []
        while node.parent != None:
            index = len(node.parent.children)
            for j in range(len(node.parent.children)):
                if node.parent.children[j].eid == node.eid:
                    index = j
                    break
            path.append(index)
            node = node.parent
        treePaths.append(list(reversed(path)))

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            if hasCausalRelationship(treePaths[i], treePaths[j]):
                candidates.discard(j)

    if len(candidates) == 0:
        raise Exception("[ERROR] No left candidate node after CasualRelationship filtering")
    else:
        max_cor = -1e10
        max_cor_index = -1
        for c in candidates:
            cor = nodes[c].confidence_score
            if cor > max_cor:
                max_cor_index = c
                max_cor = cor

    if max_cor_index == -1:
        raise Exception("[ERROR] Unable to find the node with maximal confidence score")
    else:
        return max_cor_index