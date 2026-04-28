import json
import pandas as pd
import numpy as np
import ast
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_LLM_JSON = 'dataset/raw_llm_classifications.json'
INPUT_CSV = 'dataset/skincare_db.csv'
OUTPUT_CSV = 'dataset/final_sephora_database.csv'

# Maintain a strict order for the vector indices
# Index: "acne", "blackheads", "dark_spots", "pores", "wrinkles", "eyebags"
CONDITIONS_ORDER = ["acne", "blackheads", "dark_spots", "pores", "wrinkles", "eyebags"]
MASTER_MATRIX = {
    "retinol": ["salicylic_acid", "glycolic_acid", "lactic_acid", "benzoyl_peroxide", "tretinoin", "adapalene"],
    "retinal": ["salicylic_acid", "glycolic_acid", "lactic_acid", "benzoyl_peroxide", "tretinoin", "adapalene"],
    "tretinoin": ["salicylic_acid", "glycolic_acid", "lactic_acid", "benzoyl_peroxide", "retinol", "retinal"],
    "adapalene": ["salicylic_acid", "glycolic_acid", "lactic_acid", "benzoyl_peroxide"],
    "ascorbic_acid": ["benzoyl_peroxide", "copper_tripeptide-1", "glycolic_acid", "lactic_acid", "salicylic_acid", "hydroquinone"],
    "benzoyl_peroxide": ["retinol", "tretinoin", "ascorbic_acid", "adapalene", "retinal", "hydroquinone"],
    "salicylic_acid": ["retinol", "tretinoin", "glycolic_acid", "lactic_acid", "retinal", "adapalene"],
    "glycolic_acid": ["retinol", "tretinoin", "ascorbic_acid", "salicylic_acid", "retinal", "adapalene"],
    "lactic_acid": ["retinol", "tretinoin", "ascorbic_acid", "salicylic_acid", "retinal", "adapalene"],
    "copper_tripeptide-1": ["ascorbic_acid", "glycolic_acid", "lactic_acid", "salicylic_acid"],
    "hydroquinone": ["benzoyl_peroxide", "ascorbic_acid"]
}

SCORE_MAP = {
    "PRIMARY_TREATMENT": 8.0,
    "SECONDARY_BENEFIT": 3.0,
    "NEUTRAL": 0.0,
    "CONTRAINDICATED": -5.0
}

# ==========================================
# 2. VECTOR LOGIC
# ==========================================

def get_vector_lookup():
    if not os.path.exists(RAW_LLM_JSON):
        raise FileNotFoundError(f"Waiting for {RAW_LLM_JSON}...")
    
    with open(RAW_LLM_JSON, 'r') as f:
        llm_data = json.load(f)
    
    lookup = {}
    for ing, results in llm_data.items():
        # Create a numerical list in the exact order of CONDITIONS_ORDER
        vector = [SCORE_MAP.get(results.get(cond), 0.0) for cond in CONDITIONS_ORDER]
        lookup[ing.lower().strip()] = np.array(vector)
    return lookup

def calculate_efficacy_vector(ingredients, vector_lookup):
    """Sums the ingredient vectors into a single product efficacy vector."""
    # Start with a zero vector of length 6
    product_vector = np.zeros(len(CONDITIONS_ORDER))
    
    for ing in ingredients:
        ing_clean = ing.lower().strip()
        if ing_clean in vector_lookup:
            product_vector += vector_lookup[ing_clean]
            
    return product_vector.tolist() # Store as list in CSV for easy serialization

def find_product_conflicts(ingredients):
    """Identifies what a product CONFLICTS with."""
    conflicts = set()
    for ing in ingredients:
        normalized_ing = ing.lower().strip().replace(" ", "_")
        if normalized_ing in MASTER_MATRIX:
            conflicts.update(MASTER_MATRIX[normalized_ing])
    return list(conflicts)
def price_tier(price):
    if price<20:
        return "Budget"
    elif 20<=price<50:
        return "Mid-Range"
    elif 50<=price<100:
        return "Premium"
    else:
        return "Luxury"
# ==========================================
# 3. EXECUTION
# ==========================================

print("🚀 Building efficacy vectors...")

vector_db = get_vector_lookup()
df = pd.read_csv(INPUT_CSV)

if isinstance(df['clean_ingredient_array'].iloc[0], str):
    df['clean_ingredient_array'] = df['clean_ingredient_array'].apply(ast.literal_eval)

# Generate the single vector column
df['efficacy_vector'] = df['clean_ingredient_array'].apply(lambda x: calculate_efficacy_vector(x, vector_db))

# (Keep your existing conflict logic as well)
df['conflict_tags'] = df['clean_ingredient_array'].apply(find_product_conflicts)

df['price_tier'] = df['price_usd'].apply(price_tier)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Success! Database enriched with 'efficacy_vector' column.")

