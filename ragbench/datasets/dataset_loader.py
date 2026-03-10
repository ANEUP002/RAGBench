import json

class DataLoader:
    def __init__(self, corpus_path, queries_path, ground_truth_path):
        self.corpus_path = corpus_path
        self.queries_path = queries_path
        self.ground_truth_path = ground_truth_path
    def load_json(self,path):
        with open(path,"r") as f:
            return json.load(f)
    def load_dataset(self):
        corpus = self.load_json(self.corpus_path)
        queries = self.load_json(self.queries_path)
        ground_truth = self.load_json(self.ground_truth_path)
        return corpus, queries, ground_truth

        
    