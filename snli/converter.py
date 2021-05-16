import os
import sys
import argparse
import json
from tqdm import tqdm
import re

import torch

import stanfordnlp

def snli_format(captionID, pairID, label, sentence1, sentence2,
    q_tokenized=None, q_dep_i=None, q_dep_j=None, q_dep_type=None,
    s_tokenized=None, s_dep_i=None, s_dep_j=None, s_dep_type=None):
    return {
        "captionID": captionID,
        "pairID": pairID,
        "gold_label": label,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "q_tokenized": q_tokenized,
        "q_dep_i": q_dep_i,
        "q_dep_j": q_dep_j,
        "q_dep_type": q_dep_type,
        "s_tokenized": s_tokenized,
        "s_dep_i": s_dep_i,
        "s_dep_j": s_dep_j,
        "s_dep_type": s_dep_type
    }

labels = ['contradiction', 'entailment', 'neutral']

def convert(instances, depparse_batch_size=32, num_samples=None):
    converted_instances = []
    label_invalid_cnt = 0
    for i, instance in tqdm(enumerate(instances)):
        if instance["gold_label"] in labels:
            converted_instances.append(\
                snli_format(
                    instance["captionID"],
                    instance["pairID"],
                    instance["gold_label"],
                    instance["sentence1"],
                    instance["sentence2"],
                )
            )
        else:
            label_invalid_cnt += 1
    print('Number of invalid label', label_invalid_cnt, file=sys.stderr)
    
    print("evaluating dependency...", file=sys.stderr)
    dep_type_invalid_cnt = 0

    if num_samples is None:
        num_samples = len(converted_instances)
    for i in tqdm(range(0, num_samples, depparse_batch_size)):
        nlp_input = ""
        n_sent = 0
        for j in range(i, min(len(converted_instances), i+depparse_batch_size)):
            question = converted_instances[j]["sentence2"]
            support = converted_instances[j]["sentence1"]
            converted_instances[j]["q_tokenized"] = pattern.findall(question)
            converted_instances[j]["s_tokenized"] = pattern.findall(support)
            nlp_input += ((" ".join(converted_instances[j]["q_tokenized"])) + "\n" + \
                        " ".join(converted_instances[j]["s_tokenized"]) + "\n")
            n_sent += 2
        doc = nlp(nlp_input)
        assert len(doc.sentences) == n_sent
        for j in range(i, min(len(converted_instances), i + depparse_batch_size)):
            converted_instances[j]["q_tokenized"] = [t.text for t in doc.sentences[(j-i)*2].tokens]
            converted_instances[j]["s_tokenized"] = [t.text for t in doc.sentences[(j-i)*2+1].tokens]
            converted_instances[j]["q_dep_i"] = [None] * (len(converted_instances[j]["q_tokenized"]))
            converted_instances[j]["q_dep_j"] = [None] * (len(converted_instances[j]["q_tokenized"]))
            converted_instances[j]["q_dep_type"] = [None] * (len(converted_instances[j]["q_tokenized"]))
            converted_instances[j]["s_dep_i"] = [None] * (len(converted_instances[j]["s_tokenized"]))
            converted_instances[j]["s_dep_j"] = [None] * (len(converted_instances[j]["s_tokenized"]))
            converted_instances[j]["s_dep_type"] = [None] * (len(converted_instances[j]["s_tokenized"]))

            for idx, d in enumerate(doc.sentences[(j-i)*2].dependencies):
                if type2id.unit2id(d[1]) is None:
                    dep_type_invalid_cnt += 1
                    continue
                if d[1] == 'root':
                    converted_instances[j]["q_dep_i"][idx] = int(d[2].index) - 1
                    converted_instances[j]["q_dep_j"][idx] = int(d[2].index) - 1
                    converted_instances[j]["q_dep_type"][idx] = type2id.unit2id(d[1])    
                    continue
                converted_instances[j]["q_dep_i"][idx] = int(d[0].index) - 1
                converted_instances[j]["q_dep_j"][idx] = int(d[2].index) - 1
                converted_instances[j]["q_dep_type"][idx] = type2id.unit2id(d[1])
                idx += 1
            idx = 0
            for idx, d in enumerate(doc.sentences[(j-i)*2+1].dependencies):
                if type2id.unit2id(d[1]) is None:
                    dep_type_invalid_cnt += 1
                    continue
                if d[1] == 'root':
                    converted_instances[j]["s_dep_i"][idx] = int(d[2].index) - 1
                    converted_instances[j]["s_dep_j"][idx] = int(d[2].index) - 1
                    converted_instances[j]["s_dep_type"][idx] = type2id.unit2id(d[1])
                    continue
                converted_instances[j]["s_dep_i"][idx] = int(d[0].index) - 1
                converted_instances[j]["s_dep_j"][idx] = int(d[2].index) - 1
                converted_instances[j]["s_dep_type"][idx] = type2id.unit2id(d[1])
                idx += 1
    
    print('Number of invalid dependency type', dep_type_invalid_cnt, file=sys.stderr)
    
    return converted_instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("tar", type=str)
    parser.add_argument("--depparse_batch_size", default=128, type=int)
    args = parser.parse_args()
    print(args)

    nlp = stanfordnlp.Pipeline(tokenize_pretokenized=True)
    type2id = nlp.processors['depparse'].trainer.vocab['deprel']
    pattern = re.compile('\w+|[^\w\s]')

    with open(args.src, 'r') as f:
        instances = [json.loads(line) for line in f]
    
    converted_instances = convert(instances, 
        depparse_batch_size=args.depparse_batch_size,
        num_samples=None)
    
    print("saving at {}".format(args.tar))
    with open(args.tar, "w") as f:
        for instance in converted_instances:
            f.write(json.dumps(instance) + "\n")
    