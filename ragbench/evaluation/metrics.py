def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for doc in retrieved_k if doc in relevant_set)

    return hits / k if k > 0 else 0


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for doc in retrieved_k if doc in relevant_set)

    return hits / len(relevant_set) if relevant_set else 0


def hit_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    for doc in retrieved_k:
        if doc in relevant_set:
            return 1

    return 0


def reciprocal_rank(retrieved, relevant):
    relevant_set = set(relevant)

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1 / i

    return 0


def f1_at_k(retrieved, relevant, k):
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)