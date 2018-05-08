'''
__author__: Jiaming Shen
__description__: Generate final sentences.json and entity2id.txt from quality NP phrases and tmp sentences.json.raw
    Input: 1) sentences.json.raw.tmp, 2) np_phrases.txt
    Output: 1) sentences.json, 2) entity2id.txt

__latest_updates__: 08/25/2017
'''
import sys
import json
from collections import deque

def load_candidate_phrase2id(input_file_path):
    candidate_phrase2id = {}
    eid = 0
    with open(input_file_path, "r") as fin:
        for line in fin:
            line = line.strip()
            candidate_phrase2id[line] = eid
            eid += 1
    return candidate_phrase2id

if __name__ == '__main__':
    corpusName = sys.argv[1]
    # corpusName = "cs14confs"
    input_phrase_file_path = "../../data/"+corpusName+"/intermediate/np_phrases.txt"
    input_sentence_json_path = "../../data/"+corpusName+"/intermediate/sentences.json.raw.tmp"
    output_sentence_json_path = "../../data/"+corpusName+"/intermediate/sentences.json.raw"
    #output_entity2id_path = "../../data/"+corpusName+"/intermediate/entity2id.txt"
    candidate_phrase2id = load_candidate_phrase2id(input_phrase_file_path)
    entity2id = {}
    with open(input_sentence_json_path,"r") as fin, open(output_sentence_json_path,"w") as fout:
        cnt = 0
        for line in fin:
            cnt += 1
            if (cnt % 100000 == 0):
                print("Processed %s sentences" % cnt)

            line = line.strip()
            try:
                sentence = json.loads(line)
            except:
                print(cnt)
                continue

            if ( len(sentence['tokens']) != len(sentence['pos']) ):
                print("[ERROR] Unmatched lengths of token list and pos list in %s" % str(sentence))
                continue

            new_sentence = {}
            new_tokens = []
            new_pos = []
            entityMentions = []
            index = 0
            IN_PHRASE_FLAG = False
            START_PHRASE_INDEX = 0
            q = deque()
            for i in range(len(sentence['tokens'])):
                token = sentence['tokens'][i]
                pos = sentence['pos'][i]

                if (token == "<phrase>"): # beginning of a phrase
                    IN_PHRASE_FLAG = True
                    START_PHRASE_INDEX = index
                elif (token == "</phrase>"): # end of phrase
                    phrase_token_list = []

                    while (len(q) != 0):
                        phrase_token_list.append(q.popleft())

                    phrase = " ".join(phrase_token_list)
                    if phrase.lower() in candidate_phrase2id: # quality NP phrases
                        phrase_id = candidate_phrase2id[phrase.lower()]
                        entity2id[phrase] = phrase_id
                        entityMentions.append({
                            "text": phrase,
                            "start": START_PHRASE_INDEX,
                            "end": (index-1),
                            "type": "phrase",
                            "entityId": phrase_id
                        })
                    IN_PHRASE_FLAG = False
                else:
                    new_tokens.append(token)
                    new_pos.append(pos)
                    index += 1
                    if IN_PHRASE_FLAG:
                        q.append(token)

            # sanity checking, this is possible if the sentence is segmented in the middle of a phrase
            # if (len(q) != 0):
            #     print("[ERROR]: mismatched </phrase> in sentence: %s [line number %s]" % (str(sentence), cnt))

            new_sentence['sentId'] = sentence['sentId']
            new_sentence['articleId'] = sentence['articleId']
            new_sentence['tokens'] = new_tokens
            new_sentence['pos'] = new_pos
            new_sentence['entityMentions'] = entityMentions

            json.dump(new_sentence,fout)
            fout.write("\n")

    #with open(output_entity2id_path, "w") as fout:
    #    for k,v in entity2id.items():
    #        fout.write(k+"\t"+str(v)+"\n")
