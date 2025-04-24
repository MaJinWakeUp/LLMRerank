import os
import json
import torch
from transformers import pipeline
from time import sleep
from tqdm import tqdm
from langchain_community.document_loaders import SeleniumURLLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")


CONVERSATION1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Describe the image."},
        ],
    },
]

def llava_describe_image(processor, model, image):
    prompt = processor.apply_chat_template(CONVERSATION1, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=128)
    out_text = processor.decode(output[0], skip_special_tokens=True)
    answer = out_text.split("ASSISTANT:")[-1]
    return answer

def llava_summarize_text(processor, model, text):
    conservation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Summarize the text: {text}"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conservation, add_generation_prompt=True)
    inputs = processor(None, prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=256)
    out_text = processor.decode(output[0], skip_special_tokens=True)
    answer = out_text.split("ASSISTANT:")[-1]
    return answer

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

def load_urls(urls):
    loader = SeleniumURLLoader(urls=urls)
    docs = loader.load()
    return docs

def get_img_description(img):
    description = llava_describe_image(processor, model, img)
    return description

if __name__ == "__main__":
    demo_file = "./data/demo_refine.json"
    demo_data = json.load(open(demo_file, "r"))
    images_dir = "./images"
    gcloud_out_dir = "./gcloud_output/"

    all_data = []
    for i, item in enumerate(tqdm(demo_data[:100], desc="Processing items")):
        cur_data = {}
        cur_data["misinfo_source"] = item["misinfo_source"]
        cur_data["image_id"] = item["image_id"]
        cur_data["caption"] = item["caption"]
        cur_image_path = os.path.join(images_dir, f"{item['image_id']}.jpg")
        cur_image = Image.open(cur_image_path)
        cur_data["image_description"] = get_img_description(cur_image)

        cur_gcloud_outfile = os.path.join(gcloud_out_dir, f"{item['image_id']}.json")
        gcloud_out = json.load(open(cur_gcloud_outfile, "r"))
        pages_list = gcloud_out.get("pages_with_matching_images", [])
        urls = [page.get("url", None) for page in pages_list]
        urls = [url for url in urls if url is not None]
        initial_rank = []
        urls_content = load_urls(urls)
        for url, content in zip(urls, urls_content):
            # title + description + text
            title = content.metadata.get("title", "")
            description = content.metadata.get("description", "")
            text = content.page_content
            summation = llava_summarize_text(processor, model, title+description+text)
            initial_rank.append({
                "url": url, 
                "content": summation
            })
        cur_data["initial_rank"] = initial_rank
        all_data.append(cur_data)

        # use the first rank result as evidence for misinfo
        if len(initial_rank) > 0:
            evidence = initial_rank[0]["content"]
            misinfo_judge = llava_judge_misinfo(processor, model, cur_image, item["caption"], evidence)
            cur_data["llava_judge"] = misinfo_judge
        else:
            cur_data["llava_judge"] = "No evidence found"
        
        sleep(5)
    json.dump(all_data, open("./data/initial_rank.json", "w"), indent=2)
