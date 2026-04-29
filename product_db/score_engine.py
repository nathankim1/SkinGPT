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
    """
    Implements Hybrid Pooling:
    - Composite (Max + Min) for Acne & Blackheads
    - Max Pooling for everything else
    """
    # 1. Collect vectors for all recognized ingredients
    ing_vectors = []
    for ing in ingredients:
        ing_clean = ing.lower().strip()
        if ing_clean in vector_lookup:
            ing_vectors.append(vector_lookup[ing_clean])
    
    # If no ingredients are recognized, return a neutral zero vector
    if not ing_vectors:
        return [0.0] * len(CONDITIONS_ORDER)
    
    # 2. Convert to a 2D matrix (Rows = Ingredients, Cols = Conditions)
    matrix = np.array(ing_vectors)
    final_vector = np.zeros(len(CONDITIONS_ORDER))
    
    # 3. Apply pooling logic across each condition (column)
    for i in range(len(CONDITIONS_ORDER)):
        col_data = matrix[:, i]
        
        if i < 2: # ACNE & BLACKHEADS (Composite Pooling)
            # Net score = Strongest Positive + Strongest Negative
            final_vector[i] = np.max(col_data) + np.min(col_data)
        else: # EVERYTHING ELSE (Max Pooling)
            # Score = The single most effective ingredient
            final_vector[i] = np.max(col_data)
            
    return final_vector.tolist()

def extract_actives_and_conflicts(ingredients):
    """
    Uses substring matching to identify matrix actives hidden inside messy strings
    (e.g., 'salicylic acid 2%' -> recognized_active: 'salicylic_acid')
    """
    recognized_actives = set()
    conflict_tags = set()
    
    # Handle NaN or empty values safely
    if not isinstance(ingredients, list):
        return pd.Series([[], []])
    
    for ing in ingredients:
        ing_clean = ing.lower()
        
        # Check every known active in our Master Matrix
        for active_key, enemies in MASTER_MATRIX.items():
            # Convert 'salicylic_acid' -> 'salicylic acid' to search the raw string
            search_string = active_key.replace("_", " ") 
            
            # Substring Match!
            if search_string in ing_clean:
                recognized_actives.add(active_key)
                conflict_tags.update(enemies)
                
    return pd.Series([list(recognized_actives), list(conflict_tags)])
def price_tier(price):
    if price<20:
        return "Budget"
    elif 20<=price<50:
        return "Mid-Range"
    elif 50<=price<100:
        return "Premium"
    else:
        return "Luxury"
    
def assign_routine_step(row):
    secondary = str(row.get('secondary_category', '')).strip()
    tertiary = str(row.get('tertiary_category', '')).strip()
    
    # ---------------------------------------------------------
    # 1. CLEANSERS
    # Captures: Face Washes, Toners, Exfoliators, Makeup Removers
    # Excludes: Blotting Papers
    # ---------------------------------------------------------
    if secondary == 'Cleansers':
        if tertiary == 'Blotting Papers':
            return 'Exclude'
        return 'Cleanser'
        
    # ---------------------------------------------------------
    # 2. TREATMENTS
    # Captures: Serums, Peels, Acne Spot Treatments
    # ---------------------------------------------------------
    elif secondary == 'Treatments':
        return 'Treatment'
        
    # ---------------------------------------------------------
    # 3. MOISTURIZERS
    # Captures: Standard moisturizers, Oils, Night Creams, Essences
    # Excludes: BB & CC Creams (Makeup hybrids)
    # ---------------------------------------------------------
    elif secondary == 'Moisturizers':
        if tertiary in ['BB & CC Creams']:
            return 'Exclude'
        return 'Moisturizer'
        
    # ---------------------------------------------------------
    # 4. EXCLUSIONS (The "Noise")
    # Excludes: Eye creams, Sunscreens, Masks, Lip balms, Tools, Self-Tanners
    # ---------------------------------------------------------
    return 'Exclude'
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
df[['recognized_actives', 'conflict_tags']] = df['clean_ingredient_array'].apply(extract_actives_and_conflicts)

df['price_tier'] = df['price_usd'].apply(price_tier)
df['routine_step'] = df.apply(assign_routine_step, axis=1)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Success! Database enriched with 'efficacy_vector' column.")

