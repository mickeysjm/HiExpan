import json
import sys
intputFile = sys.argv[1]

with open(intputFile, "r") as fin:
    cnt = 0
    for line in fin:
        if cnt % 100000 == 0 and cnt != 0:
            print("Processed %d lines" % cnt)

        line = line.strip()
        sentence = json.loads(line)

        for mention in sentence["entityMentions"]:
            text = mention["text"]
            start = mention["start"]
            end = mention["end"]
            if text != " ".join(sentence["tokens"][start:end+1]):
                print("ERROR:", sentence)

        cnt += 1