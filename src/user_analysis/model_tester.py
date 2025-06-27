import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class ToxicityModelTester:
    def __init__(self, model_path, tokenizer_path=None):
        """
        Classe per testare e analizzare un modello di toxicity detection
        
        Args:
            model_path (str): Path al modello salvato
            tokenizer_path (str): Path al tokenizer (se diverso dal modello)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carica modello e tokenizer
        self.load_model()
        
    def load_model(self):
        """Carica il modello e il tokenizer"""
        print(f"Caricamento modello da: {self.model_path}")
        print(f"Device utilizzato: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Modello caricato con successo!")
        except Exception as e:
            print(f"❌ Errore nel caricamento del modello: {e}")
            raise
    
    def predict_single(self, text):
        """Predice la tossicità di un singolo testo"""
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Sposta tensori sul device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        prob_toxic = probabilities[0][1].item()
        pred_label = 1 if prob_toxic > 0.5 else 0
        
        return {
            'text': text,
            'predicted_label': pred_label,
            'toxic_probability': prob_toxic,
            'non_toxic_probability': probabilities[0][0].item()
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Predice la tossicità di una lista di testi"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Sposta tensori sul device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            for j, text in enumerate(batch_texts):
                prob_toxic = probabilities[j][1].item()
                pred_label = 1 if prob_toxic > 0.5 else 0
                
                results.append({
                    'text': text,
                    'predicted_label': pred_label,
                    'toxic_probability': prob_toxic,
                    'non_toxic_probability': probabilities[j][0].item()
                })
        
        return results
    
    def test_on_dataset(self, test_csv_path, text_column='message', label_column='label'):
        """Testa il modello su un dataset CSV"""
        print(f"Caricamento dataset di test: {test_csv_path}")
        
        df = pd.read_csv(test_csv_path)
        texts = df[text_column].astype(str).tolist()
        true_labels = df[label_column].tolist()
        
        print(f"Numero di esempi: {len(texts)}")
        
        # Predizioni
        print("Esecuzione predizioni...")
        predictions = self.predict_batch(texts)
        
        # Estrai predizioni e probabilità
        pred_labels = [p['predicted_label'] for p in predictions]
        pred_probs = [p['toxic_probability'] for p in predictions]
        
        # Calcola metriche
        metrics = self.calculate_metrics(true_labels, pred_labels, pred_probs)
        
        # Salva risultati
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'predicted_label': pred_labels,
            'toxic_probability': pred_probs
        })
        
        return {
            'metrics': metrics,
            'predictions': results_df,
            'classification_report': classification_report(true_labels, pred_labels, output_dict=True)
        }
    
    def calculate_metrics(self, true_labels, pred_labels, pred_probs):
        """Calcola tutte le metriche di performance"""
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted'),
            'recall': recall_score(true_labels, pred_labels, average='weighted'),
            'f1': f1_score(true_labels, pred_labels, average='weighted'),
            'precision_toxic': precision_score(true_labels, pred_labels, pos_label=1),
            'recall_toxic': recall_score(true_labels, pred_labels, pos_label=1),
            'f1_toxic': f1_score(true_labels, pred_labels, pos_label=1),
        }
        
        if len(np.unique(true_labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(true_labels, pred_probs)
            metrics['avg_precision'] = average_precision_score(true_labels, pred_probs)
        
        return metrics
    
    def plot_performance_metrics(self, true_labels, pred_labels, pred_probs):
        """Visualizza le metriche di performance"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        
        # ROC Curve
        if len(np.unique(true_labels)) > 1:
            fpr, tpr, _ = roc_curve(true_labels, pred_probs)
            auc = roc_auc_score(true_labels, pred_probs)
            axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            axes[0,1].plot([0, 1], [0, 1], 'k--')
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend()
        
        # Precision-Recall Curve
        if len(np.unique(true_labels)) > 1:
            precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
            avg_precision = average_precision_score(true_labels, pred_probs)
            axes[0,2].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
            axes[0,2].set_xlabel('Recall')
            axes[0,2].set_ylabel('Precision')
            axes[0,2].set_title('Precision-Recall Curve')
            axes[0,2].legend()
        
        # Distribuzione delle probabilità
        toxic_probs = [pred_probs[i] for i, label in enumerate(true_labels) if label == 1]
        non_toxic_probs = [pred_probs[i] for i, label in enumerate(true_labels) if label == 0]
        
        axes[1,0].hist(non_toxic_probs, bins=30, alpha=0.7, label='Non-Toxic', color='green')
        axes[1,0].hist(toxic_probs, bins=30, alpha=0.7, label='Toxic', color='red')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Predicted Probabilities')
        axes[1,0].legend()
        
        # Box plot delle probabilità
        data_for_box = []
        labels_for_box = []
        for i, label in enumerate(true_labels):
            data_for_box.append(pred_probs[i])
            labels_for_box.append('Toxic' if label == 1 else 'Non-Toxic')
        
        df_box = pd.DataFrame({'Probability': data_for_box, 'True_Label': labels_for_box})
        sns.boxplot(data=df_box, x='True_Label', y='Probability', ax=axes[1,1])
        axes[1,1].set_title('Probability Distribution by True Label')
        
        # Threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            pred_thresh = (np.array(pred_probs) >= threshold).astype(int)
            f1_scores.append(f1_score(true_labels, pred_thresh, pos_label=1))
            precisions.append(precision_score(true_labels, pred_thresh, pos_label=1))
            recalls.append(recall_score(true_labels, pred_thresh, pos_label=1))
        
        axes[1,2].plot(thresholds, f1_scores, label='F1-Score', marker='o')
        axes[1,2].plot(thresholds, precisions, label='Precision', marker='s')
        axes[1,2].plot(thresholds, recalls, label='Recall', marker='^')
        axes[1,2].set_xlabel('Threshold')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_title('Metrics vs Threshold')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self, results_df, show_examples=10):
        """Analizza gli errori del modello"""
        # False Positives (predetti tossici ma non tossici)
        false_positives = results_df[
            (results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)
        ].sort_values('toxic_probability', ascending=False)
        
        # False Negatives (predetti non tossici ma tossici)
        false_negatives = results_df[
            (results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)
        ].sort_values('toxic_probability', ascending=True)
        
        print("="*80)
        print("ANALISI DEGLI ERRORI")
        print("="*80)
        
        print(f"\n📊 STATISTICHE:")
        print(f"Falsi Positivi: {len(false_positives)}")
        print(f"Falsi Negativi: {len(false_negatives)}")
        print(f"Accuratezza: {accuracy_score(results_df['true_label'], results_df['predicted_label']):.4f}")
        
        print(f"\n❌ TOP {show_examples} FALSI POSITIVI (erroneamente classificati come tossici):")
        print("-" * 80)
        for idx, row in false_positives.head(show_examples).iterrows():
            print(f"Testo: {row['text'][:100]}...")
            print(f"Probabilità tossicità: {row['toxic_probability']:.4f}")
            print("-" * 40)
        
        print(f"\n❌ TOP {show_examples} FALSI NEGATIVI (mancate detection di tossicità):")
        print("-" * 80)
        for idx, row in false_negatives.head(show_examples).iterrows():
            print(f"Testo: {row['text'][:100]}...")
            print(f"Probabilità tossicità: {row['toxic_probability']:.4f}")
            print("-" * 40)
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def test_custom_examples(self):
        """Testa su esempi personalizzati"""
        test_examples = [
            # Non tossici
            "Ciao, come stai oggi?",
            "Grazie per l'aiuto!",
            "Buona giornata a tutti",
            "Mi piace questo posto",
            "Complimenti per il lavoro",
            
            # Ambigui
            "Non sono d'accordo con te",
            "Questo è sbagliato secondo me",
            "Non mi piace per niente",
            "Sei proprio sicuro?",
            "Ma cosa dici?",
            
            # Tossici
            "Sei un idiota completo",
            "Vai a morire stupido",
            "Ti odio con tutto il cuore",
            "Fatti i fatti tuoi deficiente",
            "Meriteresti di sparire",
            
            # Tossici mascherati
            "Che persona 'intelligente' che sei",
            "Complimenti per il tuo QI",
            "Sei proprio un genio eh",
            "Che bella figura che fai sempre",
        ]
        
        print("="*80)
        print("TEST SU ESEMPI PERSONALIZZATI")
        print("="*80)
        
        results = []
        for text in test_examples:
            prediction = self.predict_single(text)
            results.append(prediction)
            
            status = "🔴 TOSSICO" if prediction['predicted_label'] == 1 else "🟢 NON TOSSICO"
            confidence = prediction['toxic_probability']
            
            print(f"{status} ({confidence:.3f}) | {text}")
        
        return results
    
    def benchmark_speed(self, num_samples=100):
        """Benchmark della velocità del modello"""
        import time
        
        # Genera testi di test casuali
        test_texts = [f"Questo è un testo di test numero {i}" for i in range(num_samples)]
        
        print(f"Benchmark velocità su {num_samples} campioni...")
        
        # Test singole predizioni
        start_time = time.time()
        for text in test_texts[:10]:  # Solo 10 per il test singolo
            self.predict_single(text)
        single_time = (time.time() - start_time) / 10
        
        # Test batch
        start_time = time.time()
        self.predict_batch(test_texts)
        batch_time = time.time() - start_time
        
        print(f"⏱️  Tempo medio per predizione singola: {single_time:.4f} secondi")
        print(f"⏱️  Tempo totale per batch di {num_samples}: {batch_time:.4f} secondi")
        print(f"⏱️  Predizioni per secondo (batch): {num_samples/batch_time:.2f}")
        
        return {
            'single_prediction_time': single_time,
            'batch_time': batch_time,
            'predictions_per_second': num_samples/batch_time
        }

def main():
    """Esempio di utilizzo completo"""
    # Inizializza il tester
    model_path = "./toxicity_model"  # Cambia con il tuo path
    tester = ToxicityModelTester(model_path)
    
    # Test su esempi custom
    print("1. Test su esempi personalizzati:")
    custom_results = tester.test_custom_examples()
    
    # Benchmark velocità
    print("\n2. Benchmark velocità:")
    speed_results = tester.benchmark_speed(100)
    
    # Test su dataset (se disponibile)
    test_csv_path = "./data/processed/test.csv"
    print("\n3. Test su dataset:")
    dataset_results = tester.test_on_dataset(test_csv_path)
    # 
    # # Visualizza metriche
    true_labels = dataset_results['predictions']['true_label'].tolist()
    pred_labels = dataset_results['predictions']['predicted_label'].tolist()
    pred_probs = dataset_results['predictions']['toxic_probability'].tolist()
    # 
    print("\n4. Visualizzazione metriche:")
    tester.plot_performance_metrics(true_labels, pred_labels, pred_probs)
    # 
    print("\n5. Analisi errori:")
    error_analysis = tester.analyze_errors(dataset_results['predictions'])

if __name__ == "__main__":
    main()