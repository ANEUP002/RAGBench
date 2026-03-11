import os
import json


class BEIRLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self):

        corpus_path = os.path.join(self.dataset_path, "corpus.jsonl")
        queries_path = os.path.join(self.dataset_path, "queries.jsonl")
        qrels_path = os.path.join(self.dataset_path, "qrels", "test.tsv")

        corpus = {}
        queries = {}
        ground_truth = {}

        # load corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                corpus[data["_id"]] = data["text"]

        # load queries
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                queries[data["_id"]] = data["text"]

        # load qrels
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) == 3:
                    qid, docid, rel = parts

                elif len(parts) == 4:
                    qid, _, docid, rel = parts

                else:
                    continue

                if qid not in ground_truth:
                    ground_truth[qid] = []

                ground_truth[qid].append(docid)

        return corpus, queries, ground_truth