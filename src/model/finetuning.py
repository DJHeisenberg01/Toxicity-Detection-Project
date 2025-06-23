import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Trasformazione in torch dataset
class ToxicityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        while idx not in train_labels:  # GRANDE pezza totalmente azzardata per i KeyError che compaiono quando batch_size!=1 e inefficiente ma sembra funzionare
            idx+=1
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



if __name__ == "__main__":
    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    
    # Load data
    df = pd.read_csv("src/model/data/messages_with_toxicity_labels.csv")
    print("Data loaded")
    
    
    # Train-validation-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(df["message"], df["label"], test_size=.2, random_state=652)
    print("Data split in training, validation and testing set")
    
    
    # Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    
    # Tokenize splits
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
    print("Data splits tokenized")

    
    # Create custom torch dataset
    train_dataset = ToxicityDataset(train_encodings, train_labels)
    val_dataset = ToxicityDataset(val_encodings, val_labels)

    
    # Training setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')
    # Cambio l'output layer in modo che fornisca un unico score invece che 6 per le 6 diverse classi di tossicità del modello originale    
    model.classifier = torch.nn.Linear(in_features=768, out_features=1, bias=True)
    model.to(device)
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=16)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(3):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = batch['labels'].unsqueeze(1).float().to(device)
            #print(f"labels: {labels}\nshape: {labels.shape}\ntype: {type(labels)}")
                        
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
    model.eval()
    

    # Save
    torch.save(model.state_dict(), "model_weights.pth")
