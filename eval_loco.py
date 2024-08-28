from py_BMXGo import BMXGo
from py_BMXGo import go

import sys
import time

from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
from tqdm import tqdm

if len(sys.argv) != 3:
    print("usage: python %.py model dataset")
    sys.exit()

topk = 100
model_name = sys.argv[1]
dataset_name = f"mixedbread-ai/LoCo-{sys.argv[2]}"
split_name = "dev"

print(f"model_name={model_name}")
print(f"dataset_name={dataset_name}")
print(f"split_name={split_name}")

MODEL = BMXGo.Build
print("MODEL:", MODEL)

config = BMXGo.NewConfig("word", "english", "english")

searcher = MODEL(f"{dataset_name}-{model_name}-index", config)

# 1. index
ds_corpus = load_dataset(dataset_name, "corpus", split="corpus")
ids, docs = [], []
for obj in tqdm(ds_corpus):
    ids.append(obj["_id"])
    docs.append(obj["title"] + "\n" + obj["text"])
print("doc size:", len(ids))

s_time = time.time()
# Convert Python list of ids to Go slice
ids_slice = go.Slice_string(ids)

# Convert Python list of docs to Go slice
docs_slice = go.Slice_string(docs)

# Now use both Go slices in the AddMany method
searcher.AddMany(ids_slice, docs_slice)
print("index time:", time.time() - s_time)

# 2. prepare
ds_queries = load_dataset(dataset_name, "queries", split="queries")
id2query = {}
for obj in ds_queries:
    id2query[obj["_id"]] = obj["text"]

# 3. search
ds_default = load_dataset(dataset_name, "default", split=split_name)
queries = set()
qrels = {}
for obj in tqdm(ds_default):
    query_id = obj["query-id"]
    query = id2query[query_id]
    queries.add((query_id, query))
    if obj["score"] > 0:
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][obj["corpus-id"]] = int(obj["score"])

print("searching...")
s_time = time.time()
pred_results = {}
queries = sorted(list(queries), key=lambda x: x[0])
for query_id, query in tqdm(queries):
    # print('query:', query)
    ret = searcher.Search(query, top_k=topk)
    pred_results[query_id] = {
        doc_id: float(score) for doc_id, score in zip(ret.Keys, ret.Scores)
    }
print("search time:", time.time() - s_time)

# 4. evaluate
ndcg, map_score, recall, precision = EvaluateRetrieval.evaluate(
    qrels, pred_results, k_values=[1, 3, 5, 10, 100]
)
print("ndcg:", ndcg)
print("map:", map_score)
print("recall:", recall)
print("precision:", precision)
acc = EvaluateRetrieval.evaluate_custom(qrels, pred_results, [3, 5, 10], metric="acc")
print("acc:", acc)
