from zipfile import ZipFile
import json
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import functools

@functools.lru_cache()
def json_to_dataframe(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    with ZipFile(input_file_path, "r") as z:
        for filename in z.namelist():
            print(filename)
            with z.open(filename) as f:
                data = f.read()
                file = json.loads(data.decode("utf-8"))
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])

    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    #     ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    #     js['q_idx'] = ndx
    main = m[['id','question','context','answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

@functools.lru_cache()
def to_pandas_data(input_file_path = 'train.json.zip'):
    record_path = ['data','paragraphs','qas','answers']
    verbose = 0
    train_data = json_to_dataframe(input_file_path, record_path)
    bis = train_data.drop_duplicates(subset=["context"])
    return bis

@functools.lru_cache()
def load_corpus(file_name = 'encoding.ann'):
    f = 512
    u = AnnoyIndex(f, 'angular')
    u.load(file_name)  # super fast, will just mmap the file
    return u

@functools.lru_cache()
def import_enc(file_name = "encoding.ann"):
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    embeddings = load_corpus(file_name)
    return model, embeddings

