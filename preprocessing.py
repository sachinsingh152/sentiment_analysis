import json
import re
import pandas as pd

# ============================================
# 📌 PATHS
# ============================================

INPUT_FILE = "/home/sachinsingh/Desktop/main_complete/Desktop/bdalab/project_main/Twibot-20/train.json"
OUTPUT_FILE = "/home/sachinsingh/Desktop/main_complete/Desktop/bdalab/project_local/bert_ready_data.csv"

clean_data = []

# ============================================
# 📌 1. LOAD + EXTRACT USEFUL DATA (SAFE)
# ============================================

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

for entry in data:
    
    # Skip if tweet field missing or None
    if entry.get("tweet") is None:
        continue
    
    # Safe label extraction
    try:
        label = int(entry.get("label", 0))
    except:
        continue

    tweets = entry["tweet"]

    # Handle if tweet is string instead of list
    if isinstance(tweets, str):
        tweets = [tweets]

    for tweet in tweets:
        if not tweet:
            continue
        
        # Skip retweets
        if tweet.startswith("RT"):
            continue

        clean_data.append({
            "text": tweet,
            "label": label
        })

print("After extraction:", len(clean_data))


# ============================================
# 📌 2. TEXT CLEANING
# ============================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for d in clean_data:
    d["text"] = clean_text(d["text"])


# ============================================
# 📌 3. CONVERT TO DATAFRAME
# ============================================

df = pd.DataFrame(clean_data)

# Remove short/empty text
df = df[df["text"].str.len() > 5]

# Remove duplicates
df = df.drop_duplicates(subset=["text"])

print("After cleaning:", len(df))


# ============================================
# 📌 4. BALANCE DATASET
# ============================================

if len(df["label"].unique()) > 1:
    df = df.groupby("label").sample(n=min(df["label"].value_counts()), random_state=42)

print("Balanced dataset:", len(df))


# ============================================
# 📌 5. SAVE FILE
# ============================================

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Clean dataset saved at: {OUTPUT_FILE}")