import json
import random
import re
import os
import sys
from tqdm import tqdm

def save_jsonl(dictionaries, path, print_message=True, skip_if_exists=False):
    """save jsonl file from list of dictionaries
    """
    # if os.path.exists(path):
    #     if not skip_if_exists:
    #         raise OSError("file {} already exists".format(path))
    #     else:
    #         print("CAUTION: skip saving (file {} already exists)".format(path))
    #         return

    if print_message:
        print("saving at {}".format(path))
    with open(path, "a") as out_file:
        for instance in dictionaries:
            out_file.write(json.dumps(instance) + "\n")

def read_jsonl(path):
    with open(path, "r") as in_file:
        out = [json.loads(line) for line in in_file]

    return out

def get_label_set():
    label_set = {"SUPPORTS","REFUTES","NOT ENOUGH INFO"}
    return label_set


def load_split_trainset(dev_size:int):
    """
    Loads the full training set, splits it into preliminary train and dev set.
    This preliminary dev set is balanced.
    dev_size: size of dev set.
    """

    # load fever training data
    full_train = load_fever_train()

    positives = []
    negatives = []
    neutrals = []

    # sort dataset according to label.
    for example in full_train:
        example['id']
        label = example['label']
        if label == "SUPPORTS":
            positives.append(example)
        elif label == "REFUTES":
            negatives.append(example)
        elif label == "NOT ENOUGH INFO":
            neutrals.append(example)
        else:
            raise AssertionError("Bad label!", label)

    # shuffle examples for each label.
    random.seed(42)
    random.shuffle(positives)
    random.shuffle(negatives)
    random.shuffle(neutrals)

    # split off a preliminary dev set, balanced across each of the three classes
    size = int(dev_size/3)
    preliminary_dev_set = positives[:size] + negatives[:size] + neutrals[:size]

    # the remaining data will be the new training data
    train_set = positives[size:] + negatives[size:] + neutrals[size:]

    # shuffle order of examples
    random.shuffle(preliminary_dev_set)
    random.shuffle(train_set)

    return train_set, preliminary_dev_set


def load_fever_train(path="data/train.jsonl", howmany=999999):
    """
    Reads the Fever Training set, returns list of examples.
    howmany: how many examples to load. Useful for debugging.
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline+1 >= howmany:
                break
    return data

def load_paper_dataset(train="data/train.jsonl", 
                        dev="data/dev.jsonl",
                        test="data/test.jsonl"):
    """Reads the Fever train/dev set used on the paper.
    """
    train_ds = load_fever_train(path=train, howmany=9999999999)
    dev_ds = load_fever_train(path=dev, howmany=9999999999)
    test_ds = load_fever_train(path=test, howmany=9999999999)
    return train_ds, dev_ds, test_ds



if __name__ == "__main__":
    # load fever training data
    fever_data = load_fever_train(howmany=20)
    print(len(fever_data))

    # load train and split-off dev set of size 9999.
    train, dev = load_split_trainset(9999)
    print(len(train))
    print(len(dev))


    for sample in train[:3]:
        print(sample)

    # s = Sample(train[0])
    # print(s.__dict__)
