import os
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
dataset = Dataset.from_json("../Promos/promos.jsonl")
dataset.push_to_hub("<your-username>/WWE-Promos-Finetune", token=token)
