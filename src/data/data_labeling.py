import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm


# Funzione per stimare la tossicità
def get_toxicity_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Usa il punteggio massimo tra le 6 classi
    score = torch.sigmoid(outputs.logits)[0].max().item()
    return score


if __name__ == "__main__":
    # Carica il modello
    model_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Carica i messaggi
    df = pd.read_csv("../../data/processed/cleaned_twitch_messages.csv")
    
    # Etichetta i messaggi
    tqdm.pandas()
    df["toxicity_score"] = df["message"].progress_apply(get_toxicity_score)
    df["label"] = df["toxicity_score"].apply(lambda x: 1 if x >= 0.9 else 0 )
    
    # Salva i risultati
    os.makedirs("./model/data", exist_ok=True)
    df.to_csv("./model/data/messages_with_toxicity_labels.csv", index=False)
    print("✅ Etichettatura completata.")

    # Visualizzazione risultati
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="toxicity_score", bins=30, kde=True, color="skyblue")
    plt.title("Distribuzione dei punteggi di tossicità")
    plt.xlabel("Toxicity score")
    plt.ylabel("Numero di messaggi")
    plt.show()