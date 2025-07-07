import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Transformers imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ce = nn.CrossEntropyLoss(weight=self.alpha)

        def forward(self, logits, targets):
            ce_loss = self.ce(logits, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss

class ToxicityDetectionPipeline:
    def __init__(self, data_path, model_name="dbmdz/bert-base-italian-cased"):
        """
        Pipeline per la detection di tossicità con supporto per modelli italiani
        
        Args:
            data_path (str): Path al dataset CSV
            model_name (str): Nome del modello pre-addestrato da utilizzare
        """
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.vectorizer = None
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_and_preprocess_data(self):
        """Carica e preprocessa il dataset"""
        logger.info("Caricamento dataset...")
        
        # Carica il dataset
        df = pd.read_csv(self.data_path)
        
        # Preprocessing del testo
        df['message'] = df['message'].astype(str)
        df['message'] = df['message'].str.lower().str.strip()
        
        # Rimuovi messaggi vuoti o troppo corti
        df = df[df['message'].str.len() > 2]
        
        # Analisi del dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Distribuzione delle classi:\n{df['label'].value_counts()}")
        
        # Calcola il rapporto di sbilanciamento
        class_counts = df['label'].value_counts()
        imbalance_ratio = class_counts[0] / class_counts[1]
        logger.info(f"Rapporto di sbilanciamento: {imbalance_ratio:.2f}")
        
        self.df = df
        return df
    
    def visualize_data_distribution(self):
        """Visualizza la distribuzione dei dati"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribuzione delle classi
        self.df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Distribuzione delle Classi')
        axes[0,0].set_xlabel('Label')
        axes[0,0].set_ylabel('Count')
        
        # Distribuzione della lunghezza dei messaggi
        self.df['message_length'] = self.df['message'].str.len()
        axes[0,1].hist(self.df['message_length'], bins=50, alpha=0.7)
        axes[0,1].set_title('Distribuzione Lunghezza Messaggi')
        axes[0,1].set_xlabel('Lunghezza')
        axes[0,1].set_ylabel('Frequenza')
        
        # Distribuzione toxicity score
        axes[1,0].hist(self.df['toxicity_score'], bins=50, alpha=0.7)
        axes[1,0].set_title('Distribuzione Toxicity Score')
        axes[1,0].set_xlabel('Score')
        axes[1,0].set_ylabel('Frequenza')
        
        # Boxplot toxicity score per classe
        sns.boxplot(data=self.df, x='label', y='toxicity_score', ax=axes[1,1])
        axes[1,1].set_title('Toxicity Score per Classe')
        
        plt.tight_layout()
        plt.show()
    
    def train_baseline_models(self):
        """Addestra modelli baseline con TF-IDF"""
        logger.info("Addestramento modelli baseline...")
        
        X = self.df['message']
        y = self.df['label']
        
        # Split stratificato
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # TF-IDF Vectorization con parametri ottimizzati
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=None  # Mantieni le stop words per questo dominio
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        results = {}
        
        # 1. Logistic Regression senza bilanciamento
        logger.info("Addestramento Logistic Regression (senza bilanciamento)...")
        clf_unbalanced = LogisticRegression(max_iter=1000, random_state=42)
        clf_unbalanced.fit(X_train_tfidf, y_train)
        y_pred_unbalanced = clf_unbalanced.predict(X_test_tfidf)
        y_proba_unbalanced = clf_unbalanced.predict_proba(X_test_tfidf)[:, 1]
        
        results['unbalanced'] = {
            'predictions': y_pred_unbalanced,
            'probabilities': y_proba_unbalanced,
            'report': classification_report(y_test, y_pred_unbalanced, output_dict=True),
            'auc': roc_auc_score(y_test, y_proba_unbalanced)
        }
        
        # 2. Logistic Regression con class_weight='balanced'
        logger.info("Addestramento Logistic Regression (class_weight balanced)...")
        clf_balanced = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        )
        clf_balanced.fit(X_train_tfidf, y_train)
        y_pred_balanced = clf_balanced.predict(X_test_tfidf)
        y_proba_balanced = clf_balanced.predict_proba(X_test_tfidf)[:, 1]
        
        results['balanced'] = {
            'predictions': y_pred_balanced,
            'probabilities': y_proba_balanced,
            'report': classification_report(y_test, y_pred_balanced, output_dict=True),
            'auc': roc_auc_score(y_test, y_proba_balanced)
        }
        
        # 3. Logistic Regression con SMOTE
        logger.info("Addestramento Logistic Regression (SMOTE)...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
        
        clf_smote = LogisticRegression(max_iter=1000, random_state=42)
        clf_smote.fit(X_train_smote, y_train_smote)
        y_pred_smote = clf_smote.predict(X_test_tfidf)
        y_proba_smote = clf_smote.predict_proba(X_test_tfidf)[:, 1]
        
        results['smote'] = {
            'predictions': y_pred_smote,
            'probabilities': y_proba_smote,
            'report': classification_report(y_test, y_pred_smote, output_dict=True),
            'auc': roc_auc_score(y_test, y_proba_smote)
        }
        
        # Stampa risultati
        self._print_baseline_results(results, y_test)
        
        self.baseline_results = results
        self.y_test_baseline = y_test
        
        return results
    
    def _print_baseline_results(self, results, y_test):
        """Stampa i risultati dei modelli baseline"""
        print("\n" + "="*80)
        print("RISULTATI MODELLI BASELINE")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n--- {model_name.upper()} ---")
            print(f"AUC: {result['auc']:.4f}")
            print(f"Precision (class 1): {result['report']['1']['precision']:.4f}")
            print(f"Recall (class 1): {result['report']['1']['recall']:.4f}")
            print(f"F1-Score (class 1): {result['report']['1']['f1-score']:.4f}")
            print(f"Accuracy: {result['report']['accuracy']:.4f}")
    
    def prepare_transformer_data(self):
        """Prepara i dati per il fine-tuning del transformer"""
        logger.info("Preparazione dati per transformer...")
        
        # Split stratificato
        X = self.df['message'].values
        y = self.df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Ulteriore split per validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Inizializza tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Crea datasets
        train_dataset = self._create_dataset(X_train, y_train)
        val_dataset = self._create_dataset(X_val, y_val)
        test_dataset = self._create_dataset(X_test, y_test)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.y_test_transformer = y_test
        
        logger.info(f"Train set: {len(train_dataset)}")
        logger.info(f"Validation set: {len(val_dataset)}")
        logger.info(f"Test set: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_dataset(self, texts, labels):
        """Crea un dataset HuggingFace"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                padding=False, 
                truncation=True, 
                max_length=512
            )
        
        # Crea dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        
        # Tokenizza
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return dataset
    
    

    

    def train_transformer(self, output_dir="./toxicity_model"):
        logger.info(f"Inizio fine-tuning di {self.model_name}...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        ).to(self.device)

        # Calcola class weights in modo robusto
        labels = self.train_dataset['label']
        class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(self.device)
        logger.info(f"Class weights: {class_weights}")

        # Weighted Trainer con Focal Loss
        class WeightedFocalTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                loss_fct = FocalLoss(alpha=class_weights, gamma=2)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,                 # Aumentato numero epoche
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_ratio=0.1,
            weight_decay=0.01,                   # Ridotto un po' weight decay
            learning_rate=2e-5,                  # Learning rate un po' più alto
            max_grad_norm=1.0,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1_pos",     # Assicurati metriche coerenti
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,
            seed=42,
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            lr_scheduler_type="cosine",
            dataloader_num_workers=2,
            # Aggiungi shuffle se vuoi esplicito
            # dataloader_shuffle=True  # default True
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = WeightedFocalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        eval_results = trainer.evaluate(eval_dataset=self.test_dataset)
        logger.info(f"Risultati finali: {eval_results}")

        self.trainer = trainer
        return trainer

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)

        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)

        # Metriche per classe positiva
        precision_pos = precision_score(labels, preds, pos_label=1, zero_division=0)
        recall_pos = recall_score(labels, preds, pos_label=1, zero_division=0)
        f1_pos = f1_score(labels, preds, pos_label=1, zero_division=0)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_pos': f1_pos,
            'precision_pos': precision_pos,
            'recall_pos': recall_pos,
        }

    
    def evaluate_all_models(self):
        """Confronta tutti i modelli addestrati"""
        if not hasattr(self, 'trainer'):
            logger.error("Devi prima addestrare il modello transformer!")
            return
        
        # Predizioni del transformer
        predictions = self.trainer.predict(self.test_dataset)
        y_pred_transformer = np.argmax(predictions.predictions, axis=1)
        y_proba_transformer = torch.softmax(torch.from_numpy(predictions.predictions), dim=1)[:, 1].numpy()
        
        # Confronto risultati
        print("\n" + "="*100)
        print("CONFRONTO FINALE DEI MODELLI")
        print("="*100)
        
        models_comparison = {}
        
        # Baseline models
        for model_name, result in self.baseline_results.items():
            models_comparison[f'Baseline_{model_name}'] = {
                'f1_pos': result['report']['1']['f1-score'],
                'precision_pos': result['report']['1']['precision'],
                'recall_pos': result['report']['1']['recall'],
                'auc': result['auc']
            }
        
        # Transformer
        auc_transformer = roc_auc_score(self.y_test_transformer, y_proba_transformer)
        models_comparison['Transformer'] = {
            'f1_pos': f1_score(self.y_test_transformer, y_pred_transformer, pos_label=1),
            'precision_pos': precision_score(self.y_test_transformer, y_pred_transformer, pos_label=1),
            'recall_pos': recall_score(self.y_test_transformer, y_pred_transformer, pos_label=1),
            'auc': auc_transformer
        }
        
        # Stampa tabella comparativa
        comparison_df = pd.DataFrame(models_comparison).T
        print(comparison_df.round(4))
        
        # Trova il modello migliore
        best_model = comparison_df['f1_pos'].idxmax()
        print(f"\n🏆 Modello migliore: {best_model}")
        print(f"F1-Score (classe positiva): {comparison_df.loc[best_model, 'f1_pos']:.4f}")
        
        return comparison_df
    
    def predict_examples(self, texts, model_type='transformer'):
        """Predice su esempi di testo"""
        if model_type == 'transformer' and hasattr(self, 'trainer'):
            # Tokenizza i testi
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            )
            
            # IMPORTANTE: sposta i tensori sul device corretto
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Predizione
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**encoded)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            results = []
            for i, text in enumerate(texts):
                prob_toxic = predictions[i][1].item()
                pred_label = 1 if prob_toxic > 0.5 else 0
                results.append({
                    'text': text,
                    'predicted_label': pred_label,
                    'toxic_probability': prob_toxic
                })
            
            return results
        else:
            logger.error("Modello transformer non disponibile!")
            return []
    


# Esempio di utilizzo
if __name__ == "__main__":
    # Inizializza la pipeline
    pipeline = ToxicityDetectionPipeline(
        data_path="././data/processed/messages_labeled_detoxify.csv",
        model_name="dbmdz/bert-base-italian-cased"  # Modello italiano
    )
    
    # Carica e analizza i dati
    df = pipeline.load_and_preprocess_data()
    
    # Visualizza distribuzione
    pipeline.visualize_data_distribution()
    
    # Addestra modelli baseline
    baseline_results = pipeline.train_baseline_models()
    
    # Prepara dati per transformer
    train_ds, val_ds, test_ds = pipeline.prepare_transformer_data()
    
    # Fine-tuning transformer
    trainer = pipeline.train_transformer()
    
    # Confronta tutti i modelli
    comparison = pipeline.evaluate_all_models()
    
    # Test su esempi
    test_texts = [
        "Ciao, come stai?",
        "Sei un idiota completo",
        "Grande partita oggi!",
        "Vai a morire stupido"
    ]
    
    predictions = pipeline.predict_examples(test_texts)
    print("\n" + "="*50)
    print("PREDIZIONI SU ESEMPI:")
    print("="*50)
    for pred in predictions:
        print(f"Testo: {pred['text']}")
        print(f"Predizione: {'TOSSICO' if pred['predicted_label'] == 1 else 'NON TOSSICO'}")
        print(f"Probabilità tossicità: {pred['toxic_probability']:.4f}")
        print("-" * 30)