# LLM Rankers

Here we apply the LLM Rankers to the context manipulation misinformation detection problem, focusing on re-ranking the reverse image search results. Original README file: [LLM Rankers README](README_ORI.md).

## Folder Structure
* `data` folder contains the original misinformation data (demo_refine.json), initial ranking results from reverse image search (initial_rank.json), and LLM re-ranking results (llm_rerank.json).
* `gcloud_output` folder contains the raw data from reverse image search.
* `images` folder contains the images in misinformation.
* `llmrankers` and `Rank-R1` are original LLMRankers code.
* `gcloud_search.py` uses google cloud vision api for reverse image search.
* `summarize.py` crawls url content from website urls, and use a small model to summarize the content to <200 words. These summarized docs are used for re-ranking.
* `llm_rerank.py` use LLMRerankers to re-rank the initial results.