from py_BMXGo import BMXGo
from py_BMXGo import go

import sys
import time

import ir_datasets
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

if len(sys.argv) != 3:
    print("usage: python %.py model dataset")
    sys.exit()

topk = 100
model_name = sys.argv[1]
dataset_name = f"{sys.argv[2]}"
split_name = "dev" if dataset_name in ["mteb/msmarco"] else "test"

print(f"model_name={model_name}")
print(f"dataset_name={dataset_name}")
print(f"split_name={split_name}")

MODEL = BMXGo.Build
print("MODEL:", MODEL)

config = BMXGo.NewConfig("word", "porter", "english")

searcher = MODEL(f"{dataset_name}-{model_name}-index", config)

ds = ir_datasets.load(f"beir/{dataset_name}")
qrels = ds.qrels_dict()
# 1. index
ids, docs = [], []
for doc in tqdm(ds.docs_iter()):
    ids.append(doc.doc_id)
    docs.append(doc.text)
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
id2query = {}
for obj in ds.queries_iter():
    id2query[obj.query_id] = obj.text

# 3. search
queries = []
query_ids = []
for q_id in qrels.keys():
    query = id2query[q_id]
    query_ids.append(q_id)
    queries.append(query)

# Commented out the lines
# Print document tokens
# print("Document tokens:")
# for doc_id, doc in zip(ids, docs):
#     print(f"Doc {doc_id}:") 
#     searcher.GetTokens(doc)

# Print query tokens
# print("\nQuery tokens:")
# for query_id, query in zip(query_ids, queries):
#     print(f"Query {query_id}:")
#     searcher.GetTokens(query)

print("searching...")
s_time = time.time()
pred_results = {}
# Convert Python list of queries to Go slice
queries_slice = go.Slice_string(queries)
results = searcher.SearchMany(queries_slice, topk, 50)
for query_id, res in zip(query_ids, results):
    pred_results[query_id] = {
        doc_id: float(score) for doc_id, score in zip(res.Keys, res.Scores)
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