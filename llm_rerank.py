from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult
import json
import os
from tqdm import tqdm

data_file = "initial_rank.json"
data = json.load(open(data_file, "r"))

all_data = []
for item in tqdm(data, desc="Processing items"):
    query = item["caption"]
    inital_rank = item["initial_rank"]
    # initial_rank: [{"url": , "content": }, ...]
    if len(inital_rank) == 0:
        item["rerank"] = []
    else:
        docs = [SearchResult(docid=i, text=doc["content"], score=None) for i, doc in enumerate(inital_rank)]
        ranker = SetwiseLlmRanker(model_name_or_path='google/flan-t5-large',
                                  tokenizer_name_or_path='google/flan-t5-large',
                                  device='cuda',
                                  num_child=10,
                                  scoring='generation',
                                  method='heapsort',
                                  k=10)
        rerank = ranker.rerank(query, docs)
        
        llm_rerank = []
        for doc in rerank:
            llm_rerank.append({"url": inital_rank[doc.docid]["url"], "content": inital_rank[doc.docid]["content"]})
        item["rerank"] = llm_rerank
    all_data.append(item)

json.dump(all_data, open("llm_rerank.json", "w"), indent=2)