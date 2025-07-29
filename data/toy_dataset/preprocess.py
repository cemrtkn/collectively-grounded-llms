from docx import Document
import os
import re 
import json
from datasets import Dataset, DatasetDict
import numpy as np


data_dir = "../data/"


docx_path = data_dir + "raw/test-conversations"
json_path = data_dir + "raw/train-conversations"
allowed_keys = ["npc_name", "npc_persona", "player_name", "player_persona", "conversation"]

complete_data = []

for file in os.listdir(json_path):
    data = {}
    if file.endswith(".json"):
        with open(os.path.join(json_path, file), 'r', encoding='windows-1252') as f:
            train_data = json.load(f)
        for train_key in train_data.keys():
            key = train_key.replace(" ", "_")
            if key in allowed_keys:
                data[key] = train_data[train_key]
        
        if len(data.keys()) == len(allowed_keys):
            complete_data.append(data)

print(f"Total JSON files processed: {len(complete_data)}")
        

docs = []
for filename in os.listdir(docx_path):
    # Skip non-docx and temporary files
    if not filename.endswith(".docx") or filename.startswith("~$"):
        continue
    docs.append(filename)
    doc = Document(os.path.join(docx_path, filename))

    player_name = None
    player_persona = None
    npc_name = None
    npc_persona = None
    conversation_list = []
    conversation = None

    data = {}

    for paragraph in doc.paragraphs:

        if player_persona:
            conversation_list.append(paragraph.text)

        if paragraph.text.startswith("NPC Name:"):
            # Extract the NPC name
            npc_name = re.search(r"NPC Name:\s*(.*)", paragraph.text).group(1)
            if npc_name:
                data["npc_name"] = npc_name
        elif paragraph.text.startswith("NPC Persona:"):
            # Extract the NPC persona
            npc_persona = re.search(r"NPC Persona:\s*(.*)", paragraph.text).group(1)
            if npc_persona:
                data["npc_persona"] = npc_persona
        elif paragraph.text.startswith("Player Name:"):
            # Extract the Player name
            player_name = re.search(r"Player Name:\s*(.*)", paragraph.text).group(1)
            if player_name:
                data["player_name"] = player_name
        elif paragraph.text.startswith("Player persona:"):
            # Extract the Player persona
            player_persona = re.search(r"Player persona:\s*(.*)", paragraph.text).group(1)
            if player_persona:
                data["player_persona"] = player_persona
        
    
    conversation = "\n".join([conv_element for conv_element in conversation_list])
    if conversation:
        data["conversation"] = conversation

    if len(data.keys()) == len(allowed_keys):
        complete_data.append(data)



print("Total data collected:", len(complete_data))

n_splits = 2
print()
split_ids = np.array_split(list(range(len(complete_data))), n_splits)


split_datasets = {}

for i, ids in enumerate(split_ids):
    split_datasets[f"{i}"] = Dataset.from_list([complete_data[id] for id in ids])

dss_splits = DatasetDict(split_datasets)
print(type(dss_splits))

dss_splits.save_to_disk(data_dir + "processed/toy_dataset")