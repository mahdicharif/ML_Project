from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer


def store_data(data, name_output="encoding.ann"):
    items = data

    model = SentenceTransformer('distiluse-base-multilingual-cased')

    f = 512
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

    items = items.set_index("c_id")

    for i, row in items.iterrows():
        v = model.encode(row["context"])
        t.add_item(i, v)

    t.build(10)  # 10 trees
    t.save(name_output)

    print("Done")