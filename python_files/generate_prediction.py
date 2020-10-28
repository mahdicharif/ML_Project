import pandas as pd
from functools import reduce
from IlluinProject.read_data import import_enc
from IlluinProject.predictions import produce_prediction
from IlluinProject.read_data import to_pandas_data

print("Please enter a question")
query_text = input()

print("Please enter a filename")
file_name = input()

bis = to_pandas_data(input_file_path =  file_name)

if file_name == 'train.json.zip':
    model, embeddings = import_enc("encoding.ann")
else:
    model, embeddings = import_enc("encoding_valid.ann")

nearest = produce_prediction(query_text, model, embeddings)

print(*reduce(pd.DataFrame.append, map(lambda i: bis[bis.c_id == i], nearest[0]))["context"].tolist(), sep = '\n\n')