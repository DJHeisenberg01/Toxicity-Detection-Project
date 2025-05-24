
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split


def clean_message(msg):
    msg = msg.lower()
    msg = re.sub(r"http\S+", "", msg)
    msg = re.sub(r"@\w+", "", msg)
    msg = re.sub(r"[^a-zà-ù\s]", "", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for video_url, messages in data.items():
        for msg in messages:
            rows.append({
                'author': msg['author'],
                'message': msg['message'],
                'badges': msg.get('badges', []),
                'video_url': video_url
            })
    return pd.DataFrame(rows)

def preprocess_dataframe(df):
    df['cleaned'] = df['message'].apply(clean_message)
    return df[df['cleaned'].str.strip() != ""]


df = pd.read_csv("../data/processed/twitch_messages.csv")
df_clean = preprocess_dataframe(df)


train, temp = train_test_split(df_clean, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("../data/processed/train.csv", index=False)
val.to_csv("../data/processed/val.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)
print("Dati salvati correttamente!")