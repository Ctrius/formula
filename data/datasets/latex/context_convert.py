import argparse
import json
from collections import defaultdict
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="latex")

parser.add_argument("--window", type=int, default=2)

args = parser.parse_args()

splits = ["train_dev", "test", "dev", "train"]

for split in splits:
    if not os.path.exists(f"{args.dataset}_{split}.json"):
        continue
    rf = open(f"{args.dataset}_{split}.json", "r")
    datasets = json.load(rf)
    n_datasets = []
    # {"tokens": ["Iraq", "Blames", "Market", "Blast", "on", "Coalition", "."], "pos": ["NNP", "VBZ", "NN", "NN", "IN", "NNP", "."], "entities": [], "relations": [], "org_id": "CNN_IP_20030328.1600.07"}, 
    for i, exp in enumerate(datasets):
        ltokens = []
        rtokens = []
        k = args.window - 1
        while k >= 0:
            if i > k and exp["orig_id"] == datasets[i-k-1]["orig_id"]:
                # ltokens.extend(datasets[i-k-1]["tokens"])
                ltokens.append(datasets[i-k-1]["tokens"])
            k -= 1
        k = 1
        while k < args.window + 1:
            if i < len(datasets)-k and exp["orig_id"] == datasets[i+k]["orig_id"]:
                # rtokens.extend(datasets[i+k]["tokens"])
                rtokens.append(datasets[i+k]["tokens"])
            k += 1
        exp["ltokens"] = ltokens
        exp["rtokens"] = rtokens
        n_datasets.append(exp)
    json.dump(n_datasets, open(f"{args.dataset}_{split}_context@{args.window}.json", "w"))
