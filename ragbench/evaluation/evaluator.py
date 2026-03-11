from ragbench.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
    f1_at_k
)


class Evaluator:
    def __init__(self, ground_truth, k=3):
        self.ground_truth = ground_truth
        self.k = k

    def evaluate(self, query_ids, retrieved_docs):

        precisions = []
        recalls = []
        hits = []
        reciprocal_ranks = []

        for qid, retrieved in zip(query_ids, retrieved_docs):

            if qid not in self.ground_truth:
                continue

            relevant = self.ground_truth[qid]

            retrieved_k = retrieved[:self.k]

            intersection = set(retrieved_k) & set(relevant)

            precision = len(intersection) / len(retrieved_k)
            recall = len(intersection) / len(relevant)

            hit = 1 if len(intersection) > 0 else 0

            rr = 0
            for rank, doc in enumerate(retrieved_k, start=1):
                if doc in relevant:
                    rr = 1 / rank
                    break

            precisions.append(precision)
            recalls.append(recall)
            hits.append(hit)
            reciprocal_ranks.append(rr)

        return {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "hit_rate": sum(hits) / len(hits),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "f1": (2 * (sum(precisions) / len(precisions)) * (sum(recalls) / len(recalls))) /
                ((sum(precisions) / len(precisions)) + (sum(recalls) / len(recalls)))
        }