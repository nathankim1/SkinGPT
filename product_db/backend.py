import pandas as pd
import ollama
import json
import os
from tqdm import tqdm  # <--- THE MAGIC PROGRESS BAR

# 1. Configuration
MODEL_NAME = 'llama3.1' 
save_file = 'dataset/raw_llm_classifications.json'

# 2. Load your pristine Master List
df = pd.read_csv('dataset/master_ingredients.csv')
all_ingredients = df['ingredient'].tolist()
master_classification = {}

# --- RESUME LOGIC ---
if os.path.exists(save_file):
    with open(save_file, 'r') as f:
        try:
            master_classification = json.load(f)
            print(f"Loaded {len(master_classification)} completed ingredients from backup.")
        except json.JSONDecodeError:
            print("Save file is empty. Starting fresh.")

ingredients_to_process = [ing for ing in all_ingredients if ing not in master_classification]

if len(ingredients_to_process) == 0:
    print("All ingredients are already classified! You are done.")
    exit()
# --------------------

# 3. The Prompt Engineering
def get_prompt(ingredient_chunk):
    return f"""
    You are an expert cosmetic chemist. Evaluate the following skincare ingredients. 
    Classify their efficacy against 5 conditions: "acne", "hyperpigmentation", "dryness", "wrinkles", and "eyebags".
    
    You MUST use EXACTLY one of these four categories for every condition:
    - PRIMARY_TREATMENT
    - SECONDARY_BENEFIT
    - NEUTRAL
    - CONTRAINDICATED
    
    Respond STRICTLY with valid JSON. The keys MUST be the exact ingredient name provided.
    
    Ingredients to classify:
    {json.dumps(ingredient_chunk)}
    """

# 4. The Local Batching Loop
chunk_size = 30 
total_chunks = (len(ingredients_to_process) + chunk_size - 1) // chunk_size

print(f"\nIgniting local model for {len(ingredients_to_process)} ingredients...")

# Wrap the range() function in tqdm() to generate the progress bar
for i in tqdm(range(0, len(ingredients_to_process), chunk_size), total=total_chunks, desc="Classifying", unit="batch"):
    chunk = ingredients_to_process[i:i + chunk_size]
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': get_prompt(chunk)}],
            format='json' 
        )
        
        batch_results = json.loads(response['message']['content'])
        master_classification.update(batch_results)
        
        with open(save_file, 'w') as f:
            json.dump(master_classification, f, indent=4)
            
    except Exception as e:
        # We use tqdm.write instead of print so it doesn't break the visual bar
        tqdm.write(f"\nFailed on a chunk. Error: {e}")
        with open('dataset/failed_chunks.txt', 'a') as fail_log:
            fail_log.write(f"Failed on chunk starting with: {chunk[0]} | Error: {e}\n")

print(f"\nClassification completely finished! Master JSON is ready.")