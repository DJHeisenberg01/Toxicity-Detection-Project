
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split


def clean_message(msg):
    if not isinstance(msg, str):
        return ""

    msg = msg.lower()
    msg = re.sub(r"http\S+", "", msg)                # Rimuovi URL
    msg = re.sub(r"@\w+", "", msg)                   # Rimuovi mention
    msg = re.sub(r"[^a-zà-ù\s]", "", msg)            # Rimuovi simboli
    msg = re.sub(r"\b!?\w*prime\w*!?\b", "", msg)    # Rimuovi comandi tipo !prime
    msg = re.sub(r"\s+", " ", msg).strip()           # Spazi extra
    msg = re.sub(r"(.)\1{2,}", r"\1", msg)           # Riduci ripetizioni di lettere
    return msg


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for video_url, messages in data.items():
        for msg in messages:
            author = msg.get("author", "")
            message = msg.get("message", "")
            badges = msg.get("badges", [])
            # Filtro messaggi da bot noti (come StreamElements)
            if author.lower() in ["streamelements", "nightbot"]:
                continue
            rows.append({
                'author': author,
                'message': message,
                'badges': badges,
                'video_url': video_url
            })
    return pd.DataFrame(rows)


def preprocess_dataframe(df):
    df['cleaned'] = df['message'].apply(clean_message)
    # Rimuove righe vuote o pulite troppo brevi
    df = df[df['cleaned'].str.strip().str.len() > 1]
    return df



if __name__ == "__main__":
    # Carica dati da JSON
    df = load_json("cleaned_convert.json")
    df_clean = preprocess_dataframe(df)


    #Creazione dello splitting (da capire se farlo qui o altrove)
    '''
    train, temp = train_test_split(df_clean, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    '''

    print("Preprocessing completato e dati salvati.")
