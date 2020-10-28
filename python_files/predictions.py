

def produce_prediction(query_text, model, embeddings, top_n = 3):
    query = model.encode(query_text)
    nearest = embeddings.get_nns_by_vector(query, top_n, include_distances = True)
    return nearest
