import os
import json
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from time import sleep
from tqdm import tqdm

demo_file = "/scratch/jin7/datasets/AMMeBa/demo_refine.json"
demo_data = json.load(open(demo_file, "r"))
images_dir = "/scratch/jin7/datasets/AMMeBa/images_part1/"
gcloud_out_dir = "/home/jin7/projects/misinformation/gcloud_output/"

img_save_dir = "./images/"
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

def get_url_content(url):
    # get the text content of the url
    for _ in range(3):  # Retry up to 3 times
        try:
            response = requests.get(url, headers=headers)
            break
        except Exception as e:
            print(f"Error: {e}")
            sleep(2)  # Wait for 2 seconds before retrying
    else:
        return ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return text
    else:
        return ""

summarizer = pipeline("summarization")

def llm_summary(text):
    # preprocess the text, remove any special characters
    text = text.replace('\n', ' ').replace('\r', '')
    text = ''.join(e for e in text if e.isalnum() or e.isspace())

    # Use a small LLM to summarize the document
    text = "summarize: " + text
    max_chunk_size = 1024
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = summarizer(chunks, max_length=100, min_length=20)
    summary = ' '.join([s['summary_text'] for s in summaries])
    # recursively summarize the summary until it is less than 100 words
    while len(summary.split()) > 100:
        summary = "summarize: " + summary
        max_chunk_size = 1024
        chunks = [summary[i:i + max_chunk_size] for i in range(0, len(summary), max_chunk_size)]
        summaries = summarizer(chunks, max_length=100, min_length=20)
        summary = ' '.join([s['summary_text'] for s in summaries])
    return summary

all_data = []
for i, item in enumerate(tqdm(demo_data[:20], desc="Processing items")):
    cur_data = {}
    cur_data["misinfo_source"] = item["misinfo_source"]
    cur_data["image_id"] = item["image_id"]
    cur_data["caption"] = item["caption"]
    cur_gcloud_outfile = os.path.join(gcloud_out_dir, f"{item['image_id']}.json")
    gcloud_out = json.load(open(cur_gcloud_outfile, "r"))
    pages_list = gcloud_out.get("pages_with_matching_images", [])
    initial_rank = []
    for page in pages_list:
        url = page.get("url", None)
        if url is not None:
            content = get_url_content(url)
            if content != "":
                content = llm_summary(content)
            initial_rank.append({"url": url, "content": content})
    cur_data["initial_rank"] = initial_rank
    all_data.append(cur_data)

    # save the image
    img_path = os.path.join(images_dir, f"{item['image_id']}.jpg")
    img_save_path = os.path.join(img_save_dir, f"{item['image_id']}.jpg")
    os.system(f"cp {img_path} {img_save_path}")

json.dump(all_data, open("initial_rank.json", "w"), indent=2)
