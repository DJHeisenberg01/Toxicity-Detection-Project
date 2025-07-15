import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
from detoxify import Detoxify


# Funzione per stimare la tossicità
def get_max_toxicity(text):
    try:
        scores = model.predict(text)
        return max(scores.values())  # score massimo tra tutte le classi tossiche
    except:
        return 0.0


if __name__ == "__main__":
    
   # Carica il modello
    model = Detoxify('multilingual')

    # Carica i messaggi
    df = pd.read_csv("././data/processed/cleaned_twitch_messages.csv")
    
    # Etichetta i messaggi
    tqdm.pandas()
    df["toxicity_score"] = df["message"].progress_apply(get_max_toxicity)
    df["label"] = df["toxicity_score"].apply(lambda x: 1 if x >= 0.6 else 0)
    
    # Salva i risultati
    df.to_csv("./data/processed/messages_labeled_detoxify.csv", index=False)
    print("✅ Etichettatura completata.")
    

    df = pd.read_csv("./data/processed/messages_labeled_detoxify.csv")

    # Visualizzazione risultati
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="toxicity_score", bins=30, kde=True, color="skyblue")
    plt.title("Distribuzione dei punteggi di tossicità")
    plt.xlabel("Toxicity score")
    plt.ylabel("Numero di messaggi")
    plt.show()