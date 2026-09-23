import os
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

promos_dataset = Dataset.from_json("../Promos/promos.jsonl")
promos_dataset.push_to_hub("hackSameer/WWE-Promos-Dataset", token=token)

tests_dataset = Dataset.from_json("../Promos/tests.jsonl")
tests_dataset.push_to_hub("hackSameer/WWE-Promos-Tests", token=token)

print("Both datasets uploaded successfully!")