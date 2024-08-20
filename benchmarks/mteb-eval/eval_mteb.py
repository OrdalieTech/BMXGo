import sys
import time

import ir_datasets
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

import requests

if len(sys.argv) != 3:
    print("usage: python %.py dataset")
    sys.exit()

topk = 100
model_name = sys.argv[1]
dataset_name = f"{sys.argv[2]}"
split_name = "dev" if dataset_name in ["mteb/msmarco"] else "test"

print(f"model_name={model_name}")
print(f"dataset_name={dataset_name}")
print(f"split_name={split_name}")

# Replace the MODEL definition with a function to call your Go service
def search_go_model(query, top_k):
    response = requests.post('http://localhost:8080/search', json={
        'query': query,
        'top_k': top_k
    })
    return response.json()
print("MODEL:", "Go model")

ds = ir_datasets.load(f"beir/{dataset_name}")
qrels = ds.qrels_dict()
# 1. index
ids, docs, indeces, = [], [], []
i = 0
for doc in tqdm(ds.docs_iter()):
    ids.append((doc.doc_id, i))
    docs.append(doc.text)
    indeces.append(i)
    i += 1
print("doc size:", len(ids))

s_time = time.time()
searcher.add_many(ids, docs, show_progress=True, n_workers=2)
print("index time:", time.time() - s_time)
searcher.save()

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

print("searching...")
s_time = time.time()
pred_results = {}
results = searcher.search_many(queries, top_k=topk, show_progress=True)
for query_id, res in zip(query_ids, results):
    pred_results[query_id] = {
        doc_id: float(score) for doc_id, score in zip(res.keys, res.scores)
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