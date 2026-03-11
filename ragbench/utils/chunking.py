def chunk_documents(corpus, chunk_size=256, overlap=20):
    chunks = []

    for doc_id, text in corpus.items():
        words = text.split()

        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_id}",
                "doc_id": doc_id,
                "text": chunk_text
            })

            start += chunk_size - overlap
            chunk_id += 1

    return chunks