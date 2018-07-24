"""
__author__: Jiaming Shen
__description__: Given a collection of (A,B) edges and a source node C, find the most suitable D such that
    A:B = C:D using embedding
"""
import numpy as np
from scipy.spatial import distance


def find_most_similar_fast(target_embed, embed_matrix_array, rank2eid, eid2ename, topK=1):
    """
    target_eid
    eid2embed: key = eid (int), value is the embedding of eid ( array of dimension (1, M))
    usage: find_most_similar_fast(eid2embed[114450], embed_matrix_array, rank2eid, eid2ename, topK = 3)
    """
    dist_matrix = distance.cdist(target_embed, embed_matrix_array, 'cosine')
    sorted_rank = np.argpartition(dist_matrix, topK)  # ascending order

    results = []
    for i in range(topK):
        rank = sorted_rank[0][i]
        similarity = 1.0 - dist_matrix[0][rank]
        eid = rank2eid[rank]
        results.append([eid, eid2ename[eid], similarity])

    results = sorted(results, key=lambda x: -x[2])
    # max_normalization
    max_similarity = results[0][2]
    for ele in results:
        ele[2] /= max_similarity
    return results


def find_target_embedding(seed_parent_id, seed_children_id, target_id, eid2embed, embed_dim=100):
    '''
    :param seed_parent_id:
    :param seed_children_id:
    :param target_id:
    :param eid2embed:
    :param embed_dim:
    :return:
    '''
    offset = np.zeros([1, embed_dim])
    for children_id in seed_children_id:
        offset += (eid2embed[seed_parent_id] - eid2embed[children_id])
    offset /= (len(seed_children_id))
    target_embed = eid2embed[target_id] - offset
    return target_embed


def edge_expan(seed_parent_id, seed_children_ids, target_parent_id, eid2embed, embed_matrix_array, rank2eid, eid2ename,
               embed_dim=100, topK=5):
    target_embedding = find_target_embedding(seed_parent_id, seed_children_ids, target_parent_id, eid2embed, embed_dim)
    expanded_eids = find_most_similar_fast(target_embedding, embed_matrix_array, rank2eid, eid2ename, topK)
    return expanded_eids


def find_target_embedding_using_edges(reference_edges, target_id, eid2embed, embed_dim=100):
    """
    :param reference_edges: a list of (parent, child) eids
    :param target_id:
    :param eid2embed:
    :param embed_dim:
    :return:
    """
    offset = np.zeros([1, embed_dim])
    for edge in reference_edges:
        offset += (eid2embed[edge[0]] - eid2embed[edge[1]])
    offset /= (len(reference_edges))
    target_embed = eid2embed[target_id] - offset
    return target_embed


def depth_expansion(reference_edges, target_parent_id, eid2embed, embed_matrix_array, rank2eid, eid2ename,
                    embed_dim=100, topK=5):
    target_embedding = find_target_embedding_using_edges(reference_edges, target_parent_id, eid2embed, embed_dim)
    # extract topK+1 entities in case one of the most similar entity is the target_parent node itself
    expanded_eids = find_most_similar_fast(target_embedding, embed_matrix_array, rank2eid, eid2ename, topK + 1)
    res = []
    for ele in expanded_eids:
        if ele[0] != target_parent_id:
            res.append(ele)
    # in case none of the expanded node is target_parent_id, select only the topK ones
    if len(res) > topK:
        res = res[:topK]
    return res
