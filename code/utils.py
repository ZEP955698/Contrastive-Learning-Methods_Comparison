import json
import random
from sentence_transformers import InputExample

random.seed(98)


def load_data(datatype, datapath):
    if "unsup-snli" in datatype:
        return load_snli_unsup(datapath)
    if "sup-snli" in datatype:
        return load_snli_sup(datapath)
    if "unsup-sts" in datatype:
        return load_sts_unsup(datapath)
    if "sup-sts" in datatype:
        return load_sts_sup(datapath)
    raise Exception("Unsupported data type: {}".format(datatype))


def load_snli_sup(path, verbose=True):
    samples = []
    # label_dict = {"entailment": 1, "neutral": 0, "contradiction": 2}
    label_dict = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}
    with open(path) as f:
        for i in f:
            data = json.loads(i)
            samples.append(InputExample(texts=[data["sentence1"], data["sentence2"]], label=label_dict[data["gold_label"]]))
    print("The len of  snil-sup  data is {}".format(len(samples)))
    random.shuffle(samples)
    return samples


def load_snli_unsup(path):
    samples = []
    with open(path) as f:
        for i in f:
            data = json.loads(i)
            samples.append(InputExample(texts=[data["origin"], data["entailment"], data["contradiction"]]))
    print("The len of snil-unsup data is {}".format(len(samples)))
    random.shuffle(samples)
    return samples


def load_sts_sup(path, splitter="||", max_score=5.0):
    samples = []
    with open(path, "r", encoding='UTF-8') as f:
        for i in f:
            try:
                i = i.replace("\n", "")
                data = i.split(splitter)
                samples.append(InputExample(texts=[data[1], data[2]], label=int(data[3])/max_score))
            except:
                continue
    print("The len of  sts-sup  data is {}".format(len(samples)))
    random.shuffle(samples)
    return samples


def load_sts_unsup(path):
    samples = []
    with open(path) as f:
        for i in f:
            data = i.strip()
            samples.append(InputExample(texts=[data, data]))
    print("The len of sts-unsup data is {}".format(len(samples)))
    random.shuffle(samples)
    return samples

