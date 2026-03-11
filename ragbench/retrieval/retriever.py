class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def map_indices_to_doc_ids(self, indices):
        all_results = []
        for query_result in indices:
            doc_ids = []
            seen = set()

            for idx in query_result:
                if idx == -1:
                    continue
                chunk = self.chunks[idx]
                doc_id = chunk["doc_id"]

                if doc_id not in seen:
                    doc_ids.append(doc_id)
                    seen.add(doc_id) ##To prevent adding the same doc_id in the doc_ids
            all_results.append(doc_ids)
        return all_results