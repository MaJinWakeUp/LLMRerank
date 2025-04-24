import torch
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult
import json
import os
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

def llava_judge_misinfo(processor, model, image, caption, evidence):
    conservation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Judge whether the image-caption pair is misinformation or not according to evidence. The image caption is: {caption}. The evidence is: {evidence}."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conservation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=128)
    out_text = processor.decode(output[0], skip_special_tokens=True)
    answer = out_text.split("ASSISTANT:")[-1]
    return answer

def get_rerank_results(query, initial_rank):
    # initial_rank: [{"url": , "content": }, ...]
    if len(initial_rank) == 0:
        return []
    else:
        docs = [SearchResult(docid=i, text=doc["content"], score=None) for i, doc in enumerate(initial_rank)]
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
            llm_rerank.append({"url": initial_rank[doc.docid]["url"], "content": initial_rank[doc.docid]["content"]})
        return llm_rerank

if __name__ == "__main__":
    data_file = "./data/initial_rank.json"
    data = json.load(open(data_file, "r"))

    all_data = []
    for item in tqdm(data, desc="Processing items"):
        query = item["image_description"] + " " + item["caption"]
        inital_rank = item["initial_rank"]
        rerank_result = get_rerank_results(query, inital_rank)
        item["llm_rerank"] = rerank_result

        # llava judge misinformation
        image_path = os.path.join("./images", item["image_id"] + ".jpg")
        image = Image.open(image_path)
        caption = item["caption"]
        if len(rerank_result) > 0:
            evidence = rerank_result[0]["content"]
            item["llava_judge_rerank"] = llava_judge_misinfo(processor, model, image, caption, evidence)
        else:
            item["llava_judge_rerank"] = "No evidence found"

        all_data.append(item)
    json.dump(all_data, open("./data/llm_rerank.json", "w"), indent=2)