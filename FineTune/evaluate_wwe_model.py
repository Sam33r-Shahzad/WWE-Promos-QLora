import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
from peft import PeftModel
import instructions

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_NAME = "wwe-promos"
HF_USER = "hackSameer"

DATASET_FILE = "Promos/tests.jsonl"
RUN_NAME = "2026-09-22_21.01.02"
REVISION = None

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

QUANT_4_BIT = True
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

hf_token = userdata.get('HuggingFace')
login(hf_token, add_to_git_credential=True)

dataset = load_dataset("hackSameer/WWE-Promos-Tests")

test = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]

if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
  )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

if REVISION:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
else:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME)

print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")

def model_predict(item):
    messages = [
        {"role": "system", "content": instructions.SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=400)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

set_seed(42)

for i in range(len(test)):
    print(f"Prompt: {test[i]['prompt']}")
    print(f"Generated Promo:\n{model_predict(test[i])}")
    print("=" * 60)