import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

def filter_emotes(message, all_emotes):
    emotes_in_message = []
    
    # Per ogni parola nel messaggio
    for word in message.split():
        # Se la parola è un'emote
        if word in all_emotes:
            # Rimuovi l'emote dal messaggio
            message = message.replace(word, "")
            
            # Aggiungi l'emote alla lista di emotes usate in quel messaggio
            emotes_in_message.append(word) if word not in emotes_in_message else None
    
    return emotes_in_message, message
              

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
    
    # Carica il dizionario di emotes da un file JSON e inserisci tutte le emotes in un set()
    all_emotes = set()
    with open("api/all_emotes.json", 'r', encoding='utf-8') as f:
        emotes_file = json.load(f)
    
    for emote_set in emotes_file.values():
        all_emotes.update(emote_set)
    
    
    rows = []
    for messages in data.values():
        for msg in messages:
            author = msg.get("author", "")
            author_id = msg.get("author_id", "")
            message = msg.get("message", "")
            badges = msg.get("badges", [])
            
            # Filtro delle emotes
            emotes_in_msg_list, message1 = filter_emotes(message, all_emotes)
            
            # Filtro messaggi da bot noti (come StreamElements)
            if author.lower() in ["streamelements", "nightbot"]:
                continue
            rows.append({
                'author': author,
                'message': message1,
                'badges': badges,
                'emotes': emotes_in_msg_list
            })
    return pd.DataFrame(rows)


def preprocess_dataframe(df):
    df['message'] = df['message'].apply(clean_message)
    # Rimuove righe vuote o pulite troppo brevi
    df = df[df['message'].str.strip().str.len() > 1]
    return df



if __name__ == "__main__":
    # Carica dati da JSON
    df = load_json("data/raw/all_collected_data.json")

    df_clean = preprocess_dataframe(df)
    
    print(df_clean)
    
    df_clean.to_csv("data/processed/cleaned_twitch_messages.csv", index=False)
    

    print("Preprocessing completato e dati salvati.")
