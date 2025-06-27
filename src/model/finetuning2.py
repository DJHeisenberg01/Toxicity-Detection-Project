import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Custom dataset per la tossicità
class ToxicityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Caricamento dataset
    df = pd.read_csv("src/model/data/messages_with_toxicity_labels.csv")
    print("Data loaded")

    # Split train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=652
    )
    print("Data split in training, validation and testing set")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")

    # Tokenizzazione
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
    print("Data splits tokenized")

    # Creazione dataset
    train_dataset = ToxicityDataset(train_encodings, train_labels.tolist())
    val_dataset = ToxicityDataset(val_encodings, val_labels.tolist())

    # Device e modello
    device = torch.device('cuda') 
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1)
    model.config.num_labels = 1  # Assicura che la config interna sia coerente
    model.to(device)


    # Training
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model.config.num_labels = 1  # Aggiorna la config per sicurezza
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).float().to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            loss = outputs.loss
            loss.backward()
            optim.step()

    model.eval()

    # Salvataggio
    model.save_pretrained("saved_model/")
    tokenizer.save_pretrained("saved_model/")
    print("Modello e tokenizer salvati in 'saved_model/'")
