import pandas as pd
import re
import ast

def clean_ingredients(ingredient_data):
    if pd.isna(ingredient_data):
        return []
    
    # 1. The Universal Unpacker
    if isinstance(ingredient_data, str) and ingredient_data.startswith('['):
        try:
            parsed_list = ast.literal_eval(ingredient_data)
            if isinstance(parsed_list, list):
                text = ", ".join([str(item) for item in parsed_list])
            else:
                text = str(ingredient_data)
        except (ValueError, SyntaxError):
            text = str(ingredient_data)
    else:
        text = str(ingredient_data)
        
    # Pre-comma scrubbing
    text = text.replace('\\', '/') # Fix weird backslash artifacts
    text = re.split(r'(?i)(may contain|active ingredients|inactive ingredients)', text)[0]
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[\*\.\[\]\'"]', '', text)
    
    raw_list = text.split(',')
    cleaned_list = []
    
    for item in raw_list:
        clean_item = item.strip().lower()
        
        # 2. Catch nested sub-products
        if ':' in clean_item:
            clean_item = clean_item.split(':')[-1].strip()
            
        # 3. Strip leading symbols (+, -, /, %)
        clean_item = re.sub(r'^[\+\-\/\%\s]+', '', clean_item)
        
        # 4. Normalize slashes (remove spaces around them so " / " becomes "/")
        clean_item = re.sub(r'\s*/\s*', '/', clean_item)
        
        # 5. Smart Hyphen Removal (fixes "uva-ursi" but protects "1,2-hexanediol")
        clean_item = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', clean_item)
        
        # 6. Fix double spaces
        clean_item = re.sub(r'\s+', ' ', clean_item)
        
        # 7. Strip out marketing boilerplate sentences
        clean_item = re.sub(r'(?i)\s*derived from.*$', '', clean_item)
        clean_item = re.sub(r'(?i)\s*naturally derived.*$', '', clean_item)
        clean_item = re.sub(r'(?i)\s*from natural origin.*$', '', clean_item)
        clean_item = re.sub(r'(?i)\s*sustainably sourced.*$', '', clean_item)
        
        # 8. Alias Collapsing (Water & Fragrance)
        if re.match(r'^(aqua|water|eau)(/(aqua|water|eau))*$', clean_item):
            clean_item = 'water'
        if re.match(r'^(aroma|flavor|parfum|fragrance)(/(aroma|flavor|parfum|fragrance))*$', clean_item):
            clean_item = 'fragrance'
            
        clean_item = clean_item.strip()
        
        # 9. Final Validation
        word_count = len(clean_item.split())
        is_just_number = re.match(r'^[\d\.\%]+$', clean_item)
        
        # Only keep if it's > 1 char, not a standalone number, and under 7 words
        if len(clean_item) > 1 and not is_just_number and word_count <= 6:
            cleaned_list.append(clean_item)
            
    return cleaned_list

# --- Execution ---

# Load your dataset
# Adjust the filename to match whatever the Kaggle CSV is named on your machine
df = pd.read_csv('dataset/product_info.csv')

# Filter the dataframe to just Skincare (optional, but saves compute time)
# The column name for category might vary slightly depending on the exact dataset version
if 'primary_category' in df.columns:
    skincare_df = df[df['primary_category'].str.contains('Skincare', case=False, na=False)].copy()
else:
    skincare_df = df.copy()

# Apply the scrubber
skincare_df['clean_ingredient_array'] = skincare_df['ingredients'].apply(clean_ingredients)
print(len(skincare_df))

# Let's look at a before and after for the first 5 products
print(skincare_df[['ingredients', 'clean_ingredient_array']].head())
skincare_df.to_csv('dataset/skincare_db.csv')

# 1. "Explode" the column of arrays so every single ingredient gets its own temporary row
all_ingredients_series = skincare_df['clean_ingredient_array'].explode()

# 2. Drop any empty artifacts (just in case) and extract only the unique values
unique_ingredients = all_ingredients_series.value_counts()
unique_ingredients = unique_ingredients[unique_ingredients>=3].index
master_ingredient_list = sorted(list(unique_ingredients))

# Check how many unique ingredients we actually have to send to the LLM
print(f"Total unique ingredients to classify: {len(master_ingredient_list)}")

# Optional: Save this list locally so you don't have to re-run the scrubber
pd.Series(master_ingredient_list).to_csv('dataset/master_ingredients.csv', index=False, header=['ingredient'])