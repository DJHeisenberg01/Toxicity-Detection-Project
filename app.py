import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
import re
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional

class ToxicityAnalyzer:
    def __init__(self, model_path: str):
        """
        Inizializza l'analizzatore di tossicità
        
        Args:
            model_path: Percorso al modello addestrato
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {self.device}")
        
        # Carica il tokenizer e il modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_toxicity(self, text: str) -> float:
        """
        Predice la tossicità di un singolo messaggio
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Probabilità di tossicità (0-1)
        """
        if not text or not text.strip():
            return 0.0
        
        # Tokenizza il testo
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Sposta i tensori sul device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predizione
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            # Assumendo che l'indice 1 corrisponda alla classe "tossico"
            toxicity_prob = probabilities[0][1].item()
        
        return toxicity_prob
    
    def analyze_single_message(self, message: str) -> Dict:
        """
        Analizza un singolo messaggio
        
        Args:
            message: Messaggio da analizzare
            
        Returns:
            Dizionario con risultati dell'analisi
        """
        if not message or not message.strip():
            return {
                "message": "Messaggio vuoto",
                "toxicity_score": 0.0,
                "toxicity_percentage": "0.00%",
                "classification": "Non tossico"
            }
        
        toxicity_score = self.predict_toxicity(message)
        toxicity_percentage = f"{toxicity_score * 100:.2f}%"
        
        # Classifica il messaggio
        if toxicity_score < 0.3:
            classification = "Non tossico"
        elif toxicity_score < 0.7:
            classification = "Moderatamente tossico"
        else:
            classification = "Altamente tossico"
        
        return {
            "message": message,
            "toxicity_score": toxicity_score,
            "toxicity_percentage": toxicity_percentage,
            "classification": classification
        }
    
    def parse_chat_file(self, file_content: str) -> List[Dict]:
        """
        Parsifica il contenuto di un file di chat
        
        Args:
            file_content: Contenuto del file di chat
            
        Returns:
            Lista di messaggi con utente e testo
        """
        messages = []
        lines = file_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Prova diversi formati di chat
            formats = [
                r'^([^:]+):\s*(.+)$',  # Nome: Messaggio
                r'^\[([^\]]+)\]\s*(.+)$',  # [Nome] Messaggio
                r'^<([^>]+)>\s*(.+)$',  # <Nome> Messaggio
                r'^(\w+)\s+(.+)$',  # Nome Messaggio
            ]
            
            parsed = False
            for pattern in formats:
                match = re.match(pattern, line)
                if match:
                    username = match.group(1).strip()
                    message = match.group(2).strip()
                    if username and message:
                        messages.append({
                            'username': username,
                            'message': message
                        })
                        parsed = True
                        break
            
            # Se nessun formato funziona, tratta come messaggio anonimo
            if not parsed and line:
                messages.append({
                    'username': 'Anonimo',
                    'message': line
                })
        
        return messages
    
    def analyze_chat_file(self, file_path: str) -> Dict:
        """
        Analizza un intero file di chat
        
        Args:
            file_path: Percorso al file di chat
            
        Returns:
            Dizionario con analisi completa della chat
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Errore nella lettura del file: {str(e)}"}
        
        messages = self.parse_chat_file(content)
        
        if not messages:
            return {"error": "Nessun messaggio valido trovato nel file"}
        
        # Analizza ogni messaggio
        user_toxicity = defaultdict(list)
        all_messages_analysis = []
        
        for msg in messages:
            toxicity_score = self.predict_toxicity(msg['message'])
            
            analysis = {
                'username': msg['username'],
                'message': msg['message'],
                'toxicity_score': toxicity_score,
                'toxicity_percentage': f"{toxicity_score * 100:.2f}%"
            }
            
            all_messages_analysis.append(analysis)
            user_toxicity[msg['username']].append(toxicity_score)
        
        # Calcola statistiche per utente
        user_stats = {}
        for username, scores in user_toxicity.items():
            user_stats[username] = {
                'avg_toxicity': np.mean(scores),
                'max_toxicity': np.max(scores),
                'message_count': len(scores),
                'toxic_messages': sum(1 for score in scores if score > 0.5)
            }
        
        # Top utenti più tossici (per media)
        top_toxic_users = sorted(
            user_stats.items(),
            key=lambda x: x[1]['avg_toxicity'],
            reverse=True
        )[:10]
        
        # Top messaggi più tossici
        top_toxic_messages = sorted(
            all_messages_analysis,
            key=lambda x: x['toxicity_score'],
            reverse=True
        )[:10]
        
        # Tossicità generale della chat
        all_scores = [msg['toxicity_score'] for msg in all_messages_analysis]
        overall_toxicity = np.mean(all_scores)
        
        return {
            "total_messages": len(messages),
            "total_users": len(user_stats),
            "overall_toxicity": overall_toxicity,
            "overall_toxicity_percentage": f"{overall_toxicity * 100:.2f}%",
            "top_toxic_users": top_toxic_users,
            "top_toxic_messages": top_toxic_messages,
            "user_stats": user_stats
        }

# Inizializza l'analizzatore (modifica il percorso del modello)
MODEL_PATH = "./toxicity_model"  # Modifica con il tuo percorso
analyzer = ToxicityAnalyzer(MODEL_PATH)

def analyze_single_text(message: str) -> str:
    """Interfaccia Gradio per analisi singolo messaggio"""
    if not message:
        return "Inserisci un messaggio da analizzare"
    
    result = analyzer.analyze_single_message(message)
    
    return f"""
## Analisi Messaggio

**Messaggio:** {result['message']}

**Punteggio Tossicità:** {result['toxicity_percentage']}

**Classificazione:** {result['classification']}

---
*Punteggio: 0-30% = Non tossico | 30-70% = Moderatamente tossico | 70-100% = Altamente tossico*
"""

def analyze_chat_file_interface(file) -> str:
    """Interfaccia Gradio per analisi file chat"""
    if file is None:
        return "Carica un file di chat per l'analisi"
    
    try:
        result = analyzer.analyze_chat_file(file.name)
        
        if "error" in result:
            return f"❌ **Errore:** {result['error']}"
        
        # Formatta i risultati
        output = f"""
## 📊 Analisi Chat Completa

### Statistiche Generali
- **Messaggi totali:** {result['total_messages']}
- **Utenti totali:** {result['total_users']}
- **Tossicità generale:** {result['overall_toxicity_percentage']}

### 🔥 Top 5 Utenti Più Tossici
"""
        
        for i, (username, stats) in enumerate(result['top_toxic_users'][:5], 1):
            output += f"""
**{i}. {username}**
- Tossicità media: {stats['avg_toxicity'] * 100:.2f}%
- Messaggi: {stats['message_count']}
- Messaggi tossici: {stats['toxic_messages']}
"""
        
        output += "\n### ⚠️ Top 5 Messaggi Più Tossici\n"
        
        for i, msg in enumerate(result['top_toxic_messages'][:5], 1):
            output += f"""
**{i}. {msg['username']}** ({msg['toxicity_percentage']})
*"{msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}"*
"""
        
        return output
        
    except Exception as e:
        return f"❌ **Errore nell'analisi:** {str(e)}"

# Crea l'interfaccia Gradio
def create_interface():
    with gr.Blocks(title="Analizzatore Tossicità", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# 🛡️ Analizzatore di Tossicità")
        gr.Markdown("Analizza messaggi singoli o interi file di chat per rilevare contenuti tossici")
        
        with gr.Tabs():
            # Tab per messaggio singolo
            with gr.TabItem("📝 Messaggio Singolo"):
                gr.Markdown("### Analizza un singolo messaggio")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Messaggio da analizzare",
                            placeholder="Inserisci qui il messaggio...",
                            lines=3
                        )
                        analyze_btn = gr.Button("🔍 Analizza", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Markdown(label="Risultati")
                
                analyze_btn.click(
                    fn=analyze_single_text,
                    inputs=text_input,
                    outputs=text_output
                )
                
                # Esempi
                gr.Examples(
                    examples=[
                        ["Ciao, come stai?"],
                        ["Sei proprio stupido!"],
                        ["Questo è il messaggio più bello del mondo"],
                    ],
                    inputs=text_input
                )
            
            # Tab per file chat
            with gr.TabItem("💬 File Chat"):
                gr.Markdown("### Analizza un intero file di chat")
                gr.Markdown("""
                **Formati supportati:**
                - `Nome: Messaggio`
                - `[Nome] Messaggio`
                - `<Nome> Messaggio`
                - `Nome Messaggio`
                
                Un messaggio per riga nel file di testo.
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Carica file chat (.txt)",
                            file_types=[".txt"]
                        )
                        analyze_file_btn = gr.Button("📊 Analizza Chat", variant="primary")
                    
                    with gr.Column():
                        file_output = gr.Markdown(label="Risultati Chat")
                
                analyze_file_btn.click(
                    fn=analyze_chat_file_interface,
                    inputs=file_input,
                    outputs=file_output
                )
        
        gr.Markdown("""
        ---
        ### ℹ️ Informazioni
        - Il modello restituisce una probabilità di tossicità da 0% a 100%
        - I risultati sono basati sul modello addestrato e potrebbero non essere perfetti
        - Per file di chat, vengono mostrati i top utenti e messaggi più tossici
        """)
    
    return interface

if __name__ == "__main__":
    # Crea e lancia l'interfaccia
    interface = create_interface()
    interface.launch(
        share=True,  # Crea un link pubblico
        server_name="0.0.0.0",  # Accessibile da qualsiasi IP
        server_port=7860  # Porta predefinita
    )