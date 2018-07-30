"""
__author__: Ellen Wu, Jiaming Shen
__description__: Python Implementation of SetExpan algorithm
"""
import random
import math
import time
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
# SAMPLE_RATE is the sample rate of context features. 0.8 means 80% of core skipgram features will be selected in
# each ranking pass. in HiExpan, it is set to 1
SAMPLE_RATE = 0.99
# TOP_K_SG is the maximum number of skipgrams that wil be selected to calculate the entity-entity distributional similarity.
TOP_K_SG = 200
# TOP_K_SG is the maximum number of types that wil be selected to calculate the entity-entity distributional similarity.
TOP_K_TYPE = 50
# TOP_K_EID is the number of candidate entities that we considered to calculate mrr score during each ranking pass.
TOP_K_EID = 30
# MAX_ITER_SET is the maximum number of expansion iterations, in HiExpan, it is set to 1
MAX_ITER_SET = 5
# SAMPLES is the ensemble number, in HiExpan, it is set to 1
SAMPLES = 1
# THRES_MRR is the threshold that determines whether a new entity will be included in the set or not
THRES_MRR = SAMPLES * (1.0 / 5.0)
# Skipgrams with score >= (THRESHOLD * nOfSeedEids) will be retained
THRESHOLD = 0.0
# MAXIMUM_EXPAND is maximum number of eids added into the set in each iteration
MAXIMUM_EXPAND = 10
# TOP_EID_EACH_FEATURE is the number of top eid each feature rank list will consider
TOP_EID_EACH_FEATURE = 50
# Skipgrams that can cover [FLAGS_SG_POPULARITY_LOWER, FLAGS_SG_POPULARITY_UPPER] numbers of entities will be retained
FLAGS_SG_POPULARITY_LOWER = 3
FLAGS_SG_POPULARITY_UPPER = 30 


def getSampledCoreSkipgrams(coreSkipgrams):
    sampledCoreSkipgrams = []
    for sg in coreSkipgrams:
        if random.random() <= SAMPLE_RATE:
            sampledCoreSkipgrams.append(sg)
    return sampledCoreSkipgrams


def getCombinedWeightByFeatureMap(seedEids, featuresByEidMap, weightByEidAndFeatureMap):
    combinedWeightByFeatureMap = {}
    for seed in seedEids:
        featuresOfSeed = featuresByEidMap[seed]
        for sg in featuresOfSeed:
            if sg in combinedWeightByFeatureMap:
                combinedWeightByFeatureMap[sg] += weightByEidAndFeatureMap[(seed, sg)]
            else:
                combinedWeightByFeatureMap[sg] = weightByEidAndFeatureMap[(seed, sg)]

    return combinedWeightByFeatureMap


def getFeatureSim(eid, seed, weightByEidAndFeatureMap, features):
    simWithSeed = [0.0, 0.0]
    for f in features:
        if (eid, f) in weightByEidAndFeatureMap:
            # weight_eid = weightByEidAndFeatureMap[(eid, f)]
            weight_eid = max(0.0, weightByEidAndFeatureMap[(eid, f)])
        else:
            weight_eid = 0.0
        if (seed, f) in weightByEidAndFeatureMap:
            # weight_seed = weightByEidAndFeatureMap[(seed, f)]
            weight_seed = max(0.0, weightByEidAndFeatureMap[(seed, f)])
        else:
            weight_seed = 0.0
        # Weighted Jaccard similarity
        simWithSeed[0] += min(weight_eid, weight_seed)
        simWithSeed[1] += max(weight_eid, weight_seed)
    if simWithSeed[1] == 0:
        res = 0.0
    else:
        res = simWithSeed[0]*1.0/simWithSeed[1]
    return res


def sim_sib(eid1, eid2, eid2patterns, pattern2eids, eidAndPattern2strength, eid2embed, eid2types,
            eidAndType2strength, topK_quality_sg=150):
    #  skipgram-similarity
    skipgram_features = getCombinedWeightByFeatureMap([eid1, eid2], eid2patterns, eidAndPattern2strength)

    redundantSkipgrams = set()
    for i in skipgram_features:
        size = len(pattern2eids[i])
    if size < FLAGS_SG_POPULARITY_LOWER or size > FLAGS_SG_POPULARITY_UPPER:
        redundantSkipgrams.add(i)
    for sg in redundantSkipgrams:
        del skipgram_features[sg]

    if topK_quality_sg < 0:
        quality_skipgram_features = skipgram_features
    else:
        quality_skipgram_features = {}
        for ele in sorted(skipgram_features.items(), key=lambda x: -x[1])[0:topK_quality_sg]:
            quality_skipgram_features[ele[0]] = ele[1]

    skipgram_sim = max(getFeatureSim(eid1, eid2, eidAndPattern2strength, quality_skipgram_features), 0.0)

    # embedding-similarity
    if (eid1 not in eid2embed) or (eid2 not in eid2embed):
        embedding_sim = 0.0
    else:
        embedding_sim = float(1.0 - distance.cdist(eid2embed[eid1], eid2embed[eid2], 'cosine'))
    embedding_sim = max(embedding_sim, 0.0)

    # type-similarity
    type_features = getCombinedWeightByFeatureMap([eid1, eid2], eid2types, eidAndType2strength)
    type_sim = getFeatureSim(eid1, eid2, eidAndType2strength, type_features)
    type_sim = max(type_sim, 0.0)

    overall_sim = math.sqrt(embedding_sim * (1.0 + skipgram_sim) * (1.0 + type_sim))
    return overall_sim


def sim_sib_embed_only(eid1, eid2, eid2embed):
    if (eid1 not in eid2embed) or (eid2 not in eid2embed):
        return 0.0

    embedding_sim = float(1.0 - distance.cdist(eid2embed[eid1], eid2embed[eid2], 'cosine'))
    embedding_sim = max(embedding_sim, 0.0)
    return embedding_sim


def sim_sib_skipgram_only(eid1, eid2, eid2patterns, pattern2eids, eidAndPattern2strength, topK_quality_sg=-1):
    skipgram_features = getCombinedWeightByFeatureMap([eid1, eid2], eid2patterns, eidAndPattern2strength)

    redundantSkipgrams = set()
    for i in skipgram_features:
        size = len(pattern2eids[i])
    if size < FLAGS_SG_POPULARITY_LOWER or size > FLAGS_SG_POPULARITY_UPPER:
        redundantSkipgrams.add(i)
    for sg in redundantSkipgrams:
        del skipgram_features[sg]

    if topK_quality_sg < 0:
        quality_skipgram_features = skipgram_features
    else:
        quality_skipgram_features = {}
        for ele in sorted(skipgram_features.items(), key=lambda x: -x[1])[0:topK_quality_sg]:
            quality_skipgram_features[ele[0]] = ele[1]

    skipgram_sim = max(getFeatureSim(eid1, eid2, eidAndPattern2strength, quality_skipgram_features), 0.0)
    return skipgram_sim


def sim_sib_type_only(eid1, eid2, eid2types, eidAndType2strength):
    type_features = getCombinedWeightByFeatureMap([eid1, eid2], eid2types, eidAndType2strength)
    type_sim = max(getFeatureSim(eid1, eid2, eidAndType2strength, type_features), 0.0)
    return type_sim

# expand the set of seedEntities and return eids by order, excluding seedEntities (original children)
def setExpan(seedEidsWithConfidence, negativeSeedEids, eid2patterns, pattern2eids, eidAndPattern2strength,
                         eid2types, type2eids, eidAndType2strength, eid2ename, eid2embed,
                         source_weights={"sg":1.0, "tp":5.0, "eb":1.0}, max_expand_eids=5,
                         use_embed=False, use_type=False, FLAGS_VERBOSE=False, FLAGS_DEBUG=False, ):
    """

    :param seedEidsWithConfidence: a list of [eid (int), confidence_score (float)]
    :param negativeSeedEids: a set of eids (int) that should not be included
    :param eid2patterns:
    :param pattern2eids:
    :param eidAndPattern2strength:
    :param eid2types:
    :param type2eids:
    :param eidAndType2strength:
    :param eid2ename:

    :return: a list of expanded [eid (excluding the original input eids in seedEids), confidence_score]
    """

    MAXIMUM_EXPAND = max_expand_eids
    seedEids = [ele[0] for ele in seedEidsWithConfidence]
    eid2confidence = {ele[0]: ele[1] for ele in seedEidsWithConfidence}

    # Cache the seedEids for later use
    cached_seedEids = set([ele for ele in seedEids])
    if FLAGS_VERBOSE:
        print('Seed set:')
        for eid in seedEids:
            print(eid, eid2ename[eid])
        print("[INFO] Start SetExpan")

    iters = 0
    while iters < MAX_ITER_SET:
        iters += 1
        prev_seeds = set(seedEids)
        start = time.time()
        nOfSeedEids = len(seedEids)

        ### Skipgram similarity
        combinedWeightBySkipgramMap = getCombinedWeightByFeatureMap(seedEids, eid2patterns, eidAndPattern2strength)

        # pruning skipgrams which can match too few or too many eids
        redundantSkipgrams = set()
        for i in combinedWeightBySkipgramMap:
            size = len(pattern2eids[i])
            if size < FLAGS_SG_POPULARITY_LOWER or size > FLAGS_SG_POPULARITY_UPPER:
                redundantSkipgrams.add(i)
        for sg in redundantSkipgrams:
            del combinedWeightBySkipgramMap[sg]

        # get final core skipgram features
        coreSkipgrams = []
        count = 0
        for sg in sorted(combinedWeightBySkipgramMap, key=combinedWeightBySkipgramMap.__getitem__, reverse=True):
            if count >= TOP_K_SG:
                break
            count += 1
            if combinedWeightBySkipgramMap[sg] * 1.0 / nOfSeedEids > THRESHOLD:
                coreSkipgrams.append(sg)

        # Type similarity
        if use_type:
            combinedWeightByTypeMap = getCombinedWeightByFeatureMap(seedEids, eid2types, eidAndType2strength)
            coreTypes = sorted(combinedWeightByTypeMap, key=combinedWeightByTypeMap.__getitem__, reverse=True)[0:TOP_K_TYPE]

        end = time.time()
        if FLAGS_DEBUG:
            if use_type:
                print("[INFO] CoreTypes of the seedsEids at iteration %s is %s" % (iters, coreTypes[0:5]))
            print("[INFO] Finish context feature selection using time %s seconds" % (end - start))

        # rank ensemble of skipgram similarity
        all_start = time.time()
        eid2mrr = {}
        if FLAGS_DEBUG:
            print("Start ranking ensemble at iteration %s:" % iters, end=" ")
        for i in range(SAMPLES):
            start = time.time()
            if FLAGS_DEBUG:
                print("Ensemble batch", i, end=" ")
            sampledCoreSkipgrams = getSampledCoreSkipgrams(coreSkipgrams)
            combinedSgSimByCandidateEid = {}

            candidates = set()
            for sg in sampledCoreSkipgrams:
                candidates = candidates.union(pattern2eids[sg])

            for eid in candidates:
                combinedSgSimByCandidateEid[eid] = 0.0
                for seed in seedEids:
                    combinedSgSimByCandidateEid[eid] += getFeatureSim(eid, seed, eidAndPattern2strength, sampledCoreSkipgrams)

            # get top k candidates
            count = 0
            for eid in sorted(combinedSgSimByCandidateEid, key=combinedSgSimByCandidateEid.__getitem__, reverse=True):
                if count >= TOP_K_EID:
                    break
                if eid not in seedEids:
                    count += 1
                    if eid in eid2mrr:
                        eid2mrr[eid] += 1.0 / count
                    else:
                        eid2mrr[eid] = 1.0 / count

            end = time.time()
            if FLAGS_DEBUG:
                print("using time %s" % (end - start), end=" ")

        topSkipgramEids = sorted(eid2mrr.items(), key=lambda x: -x[1])
        topSkipgramEids = topSkipgramEids[:min([TOP_EID_EACH_FEATURE, len(topSkipgramEids)])]

        if use_type:
            combinedTypeSimByCandidateEid = {}
            candidates = set()
            for tp in coreTypes:
                candidates = candidates.union(type2eids[tp])
            for eid in candidates:
                if eid in seedEids:
                    continue
                combinedTypeSimByCandidateEid[eid] = 0.0
                for seed in seedEids:
                    combinedTypeSimByCandidateEid[eid] += getFeatureSim(eid, seed, eidAndType2strength, coreTypes)

            topTypeEids = sorted(combinedTypeSimByCandidateEid.items(), key=lambda x: -x[1])
            topTypeEids = topTypeEids[:min([TOP_EID_EACH_FEATURE, len(topTypeEids)])]

        if use_embed:
            combinedEmbedSimByCandidateEid = {}
            if use_type:
                candidates = set(combinedSgSimByCandidateEid.keys()).union(combinedTypeSimByCandidateEid.keys())
            else:
                candidates = set(combinedSgSimByCandidateEid.keys())
            for eid in candidates:
                if eid in seedEids:
                    continue
                combinedEmbedSimByCandidateEid[eid] = 0.0
                if eid not in eid2embed:
                    continue
                for seed in seedEids:
                    if seed not in eid2embed:
                        continue
                    combinedEmbedSimByCandidateEid[eid] += float(1.0 - distance.cdist(eid2embed[eid], eid2embed[seed], 'cosine'))

            topEmedEids = sorted(combinedEmbedSimByCandidateEid.items(), key=lambda x: -x[1])
            topEmedEids = topEmedEids[:min([TOP_EID_EACH_FEATURE, len(topEmedEids)])]

        all_end = time.time()

        if FLAGS_DEBUG:
            print("[INFO] Candidate entities matched by skipgrams = %s" % len(combinedSgSimByCandidateEid.keys()))
            print("[INFO] Candidate entities that matched by either skipgrams or types = %s" %
                        len(set(combinedSgSimByCandidateEid.keys()).union(set(combinedTypeSimByCandidateEid.keys()))))
            print("End ranking ensemble at iteration %s" % iters)
            print("Totally using time %s seconds" % (all_end - all_start))
            print("=" * 40)
            print("[INFO] Skipgram Top Results: %s" % [[eid2ename[ele[0]], ele[1]] for ele in topSkipgramEids])
            print("="*40)
            print("[INFO] Type Top Results: %s" % [[eid2ename[ele[0]], ele[1]] for ele in topTypeEids])
            print("="*40)
            print("[INFO] Embedding Top Results: %s" % [[eid2ename[ele[0]], ele[1]] for ele in topEmedEids])
            print("="*40)

        # [New, combine skipgram, type, embeddings] Select entities to be added into the set
        eid2aggregated_mrr = defaultdict(float)  # smaller aggregated_rank, better eid
        for rank, ele in enumerate(topSkipgramEids):
            eid = ele[0]
            eid2aggregated_mrr[eid] += (source_weights["sg"] * (1.0/(rank+1)) )
        if use_type:
            for rank, ele in enumerate(topTypeEids):
                eid = ele[0]
                eid2aggregated_mrr[eid] += (source_weights["tp"] * (1.0/(rank+1)) )
        if use_embed:
            for rank, ele in enumerate(topEmedEids):
                eid = ele[0]
                eid2aggregated_mrr[eid] += (source_weights["eb"] * (1.0/(rank+1)) )

        eid_incremental = []
        for ele in sorted(eid2aggregated_mrr.items(), key=lambda x:-x[1])[0:MAXIMUM_EXPAND]:
            eid = ele[0]
            if eid not in negativeSeedEids:  # directly filter negativeSeedEids
                sib_sim_list = []
                for prev_eid in seedEids:
                    sibling_similarity = sim_sib(eid, prev_eid, eid2patterns, pattern2eids, eidAndPattern2strength,
                                                 eid2embed, eid2types, eidAndType2strength, topK_quality_sg=TOP_K_SG)
                    if abs(sibling_similarity) < 1e-10:
                        sibling_similarity = 1e-10
                    if sibling_similarity > 1.0:
                        sibling_similarity = 1.0
                    sib_sim_list.append( eid2confidence[prev_eid] + math.log(sibling_similarity) )

                sib_sim_list = np.array(sib_sim_list)
                if len(sib_sim_list) < 3:
                    confidence_score = np.mean(sib_sim_list)
                else:
                    confidence_score = np.mean(sorted(sib_sim_list)[1:-1])

                if FLAGS_DEBUG:
                    print("Add entity %s with confidence score %s" % (eid2ename[eid], confidence_score))
                eid_incremental.append(eid)
                eid2confidence[eid] = confidence_score

        seedEids.extend(eid_incremental)

        if len(set(seedEids).difference(prev_seeds)) == 0 and len(prev_seeds.difference(set(seedEids))) == 0:
            print("[INFO] Terminated due to no additional quality entities at iteration %s" % iters)
            break

    expanded = []
    for eid in seedEids:
        if eid not in cached_seedEids:
            expanded.append([eid, eid2confidence[eid]])

    return expanded
