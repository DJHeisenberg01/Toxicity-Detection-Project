import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
import re
import ast
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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
    
    def parse_badges_emotes(self, badges_str: str, emotes_str: str) -> Tuple[List[str], List[str]]:
        """
        Parsifica badges ed emotes dalle stringhe
        
        Args:
            badges_str: Stringa con badges (es: "['premium']")
            emotes_str: Stringa con emotes (es: "*[]*")
            
        Returns:
            Tuple con liste di badges ed emotes
        """
        badges = []
        emotes = []
        
        # Parsifica badges
        try:
            if badges_str and badges_str.strip() != '[]':
                badges = ast.literal_eval(badges_str)
        except:
            badges = []
        
        # Parsifica emotes (rimuove * e parsifica)
        try:
            if emotes_str and emotes_str.strip() != '*[]*':
                emotes_clean = emotes_str.strip('*')
                if emotes_clean != '[]':
                    emotes = ast.literal_eval(emotes_clean)
        except:
            emotes = []
        
        return badges, emotes
    
    def analyze_csv_file(self, file_path: str) -> Dict:
        """
        Analizza un file CSV di chat
        
        Args:
            file_path: Percorso al file CSV
            
        Returns:
            Dizionario con analisi completa della chat
        """
        try:
            # Leggi il CSV
            df = pd.read_csv(file_path)
            
            # Verifica che abbia le colonne necessarie
            required_columns = ['author', 'message', 'badges', 'emotes']
            if not all(col in df.columns for col in required_columns):
                return {"error": f"Il file CSV deve contenere le colonne: {', '.join(required_columns)}"}
            
            # Rimuovi righe con messaggi vuoti
            df = df.dropna(subset=['message'])
            df = df[df['message'].str.strip() != '']
            
            if len(df) == 0:
                return {"error": "Nessun messaggio valido trovato nel file"}
            
        except Exception as e:
            return {"error": f"Errore nella lettura del file CSV: {str(e)}"}
        
        # Analizza ogni messaggio
        user_data = defaultdict(lambda: {
            'messages': [],
            'toxicity_scores': [],
            'badges': set(),
            'emotes': []
        })
        
        all_messages_analysis = []
        
        for idx, row in df.iterrows():
            author = row['author']
            message = row['message']
            badges_str = str(row['badges']) if pd.notna(row['badges']) else '[]'
            emotes_str = str(row['emotes']) if pd.notna(row['emotes']) else '*[]*'
            
            # Parsifica badges ed emotes
            badges, emotes = self.parse_badges_emotes(badges_str, emotes_str)
            
            # Analizza tossicità
            toxicity_score = self.predict_toxicity(message)
            
            analysis = {
                'author': author,
                'message': message,
                'toxicity_score': toxicity_score,
                'toxicity_percentage': f"{toxicity_score * 100:.2f}%",
                'badges': badges,
                'emotes': emotes
            }
            
            all_messages_analysis.append(analysis)
            
            # Aggiungi ai dati utente
            user_data[author]['messages'].append(message)
            user_data[author]['toxicity_scores'].append(toxicity_score)
            user_data[author]['badges'].update(badges)
            user_data[author]['emotes'].extend(emotes)
        
        # Calcola statistiche per utente
        user_stats = {}
        for username, data in user_data.items():
            scores = data['toxicity_scores']
            user_stats[username] = {
                'avg_toxicity': np.mean(scores),
                'max_toxicity': np.max(scores),
                'min_toxicity': np.min(scores),
                'message_count': len(scores),
                'toxic_messages': sum(1 for score in scores if score > 0.5),
                'badges': list(data['badges']),
                'emotes': list(set(data['emotes'])),  # Rimuovi duplicati
                'emote_count': len(data['emotes'])
            }
        
        # Top utenti più tossici
        top_toxic_users = sorted(
            user_stats.items(),
            key=lambda x: x[1]['avg_toxicity'],
            reverse=True
        )
        
        # Top messaggi più tossici
        top_toxic_messages = sorted(
            all_messages_analysis,
            key=lambda x: x['toxicity_score'],
            reverse=True
        )
        
        # Statistiche generali
        all_scores = [msg['toxicity_score'] for msg in all_messages_analysis]
        overall_toxicity = np.mean(all_scores)
        
        # Statistiche badge
        all_badges = []
        for user_data in user_stats.values():
            all_badges.extend(user_data['badges'])
        badge_counts = Counter(all_badges)
        
        # Statistiche emotes
        all_emotes = []
        for user_data in user_stats.values():
            all_emotes.extend(user_data['emotes'])
        emote_counts = Counter(all_emotes)
        
        # Distribuzione tossicità
        toxicity_distribution = {
            'non_toxic': sum(1 for score in all_scores if score < 0.3),
            'moderately_toxic': sum(1 for score in all_scores if 0.3 <= score < 0.7),
            'highly_toxic': sum(1 for score in all_scores if score >= 0.7)
        }
        
        return {
            "total_messages": len(all_messages_analysis),
            "total_users": len(user_stats),
            "overall_toxicity": overall_toxicity,
            "overall_toxicity_percentage": f"{overall_toxicity * 100:.2f}%",
            "top_toxic_users": top_toxic_users,
            "top_toxic_messages": top_toxic_messages,
            "user_stats": user_stats,
            "badge_counts": badge_counts,
            "emote_counts": emote_counts,
            "toxicity_distribution": toxicity_distribution,
            "all_scores": all_scores
        }
    
    def create_toxicity_charts(self, analysis_result: Dict) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """
        Crea grafici riassuntivi della chat
        
        Args:
            analysis_result: Risultato dell'analisi
            
        Returns:
            Tuple con i grafici
        """
        # Grafico 1: Distribuzione tossicità
        fig1 = go.Figure(data=[go.Pie(
            labels=['Non Tossico', 'Moderatamente Tossico', 'Altamente Tossico'],
            values=[
                analysis_result['toxicity_distribution']['non_toxic'],
                analysis_result['toxicity_distribution']['moderately_toxic'],
                analysis_result['toxicity_distribution']['highly_toxic']
            ],
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig1.update_layout(
            title="Distribuzione Tossicità Messaggi",
            font=dict(size=14),
            height=400
        )
        
        # Grafico 2: Top 10 utenti per tossicità media
        top_users = analysis_result['top_toxic_users'][:10]
        usernames = [user[0] for user in top_users]
        toxicity_scores = [user[1]['avg_toxicity'] * 100 for user in top_users]
        
        fig2 = go.Figure(data=[go.Bar(
            x=toxicity_scores,
            y=usernames,
            orientation='h',
            marker_color='#e74c3c',
            text=[f"{score:.1f}%" for score in toxicity_scores],
            textposition='auto'
        )])
        
        fig2.update_layout(
            title="Top 10 Utenti Più Tossici",
            xaxis_title="Tossicità Media (%)",
            yaxis_title="Utenti",
            font=dict(size=12),
            height=500
        )
        
        # Grafico 3: Istogramma distribuzione punteggi tossicità
        fig3 = go.Figure(data=[go.Histogram(
            x=analysis_result['all_scores'],
            nbinsx=20,
            marker_color='#3498db',
            opacity=0.7
        )])
        
        fig3.update_layout(
            title="Distribuzione Punteggi di Tossicità",
            xaxis_title="Punteggio Tossicità",
            yaxis_title="Numero di Messaggi",
            font=dict(size=12),
            height=400
        )
        
        return fig1, fig2, fig3

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

def format_badges_emotes(badges: List[str], emotes: List[str]) -> str:
    """Formatta badges ed emotes per la visualizzazione"""
    result = ""
    
    if badges:
        badges_str = " ".join([f"🏅 {badge}" for badge in badges])
        result += f"**Badges:** {badges_str}  \n"
    
    if emotes:
        emotes_str = " ".join([f"😀 {emote}" for emote in emotes[:10]])  # Limita a 10 emotes
        if len(emotes) > 10:
            emotes_str += f" ... (+{len(emotes) - 10} altre)"
        result += f"**Emotes:** {emotes_str}  \n"
    
    return result

def analyze_csv_file_interface(file) -> Tuple[str, go.Figure, go.Figure, go.Figure]:
    """Interfaccia Gradio per analisi file CSV"""
    if file is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Nessun dato disponibile")
        return "Carica un file CSV per l'analisi", empty_fig, empty_fig, empty_fig
    
    try:
        result = analyzer.analyze_csv_file(file.name)
        
        if "error" in result:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Errore nei dati")
            return f"❌ **Errore:** {result['error']}", empty_fig, empty_fig, empty_fig
        
        # Crea i grafici
        fig1, fig2, fig3 = analyzer.create_toxicity_charts(result)
        
        # Formatta i risultati
        output = f"""
## 📊 Analisi Chat Completa

### Statistiche Generali
- **Messaggi totali:** {result['total_messages']:,}
- **Utenti totali:** {result['total_users']:,}
- **Tossicità generale:** {result['overall_toxicity_percentage']}
- **Messaggi non tossici:** {result['toxicity_distribution']['non_toxic']:,}
- **Messaggi moderatamente tossici:** {result['toxicity_distribution']['moderately_toxic']:,}
- **Messaggi altamente tossici:** {result['toxicity_distribution']['highly_toxic']:,}

### 🏅 Badges più Comuni
"""
        
        if result['badge_counts']:
            for badge, count in result['badge_counts'].most_common(5):
                output += f"- **{badge}:** {count:,} utenti\n"
        else:
            output += "- Nessun badge trovato\n"
        
        output += "\n### 😀 Emotes più Utilizzate\n"
        
        if result['emote_counts']:
            for emote, count in result['emote_counts'].most_common(10):
                output += f"- **{emote}:** {count:,} volte\n"
        else:
            output += "- Nessuna emote trovata\n"
        
        output += "\n### 🔥 Top 10 Utenti Più Tossici\n"
        
        for i, (username, stats) in enumerate(result['top_toxic_users'][:10], 1):
            badges_emotes = format_badges_emotes(stats['badges'], stats['emotes'])
            output += f"""
**{i}. {username}**
- **Tossicità media:** {stats['avg_toxicity'] * 100:.2f}%
- **Messaggi:** {stats['message_count']:,}
- **Messaggi tossici:** {stats['toxic_messages']:,}
{badges_emotes}
"""
        
        output += "\n### ⚠️ Top 10 Messaggi Più Tossici\n"
        
        for i, msg in enumerate(result['top_toxic_messages'][:10], 1):
            badges_emotes = format_badges_emotes(msg['badges'], msg['emotes'])
            message_preview = msg['message'][:150] + ('...' if len(msg['message']) > 150 else '')
            output += f"""
**{i}. {msg['author']}** ({msg['toxicity_percentage']})
{badges_emotes}
*"{message_preview}"*

"""
        
        return output, fig1, fig2, fig3
        
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Errore nell'analisi")
        return f"❌ **Errore nell'analisi:** {str(e)}", empty_fig, empty_fig, empty_fig

# Crea l'interfaccia Gradio
def create_interface():
    with gr.Blocks(title="Analizzatore Tossicità", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# 🛡️ Analizzatore di Tossicità Chat")
        gr.Markdown("Analizza messaggi singoli o file CSV di chat per rilevare contenuti tossici con visualizzazioni avanzate")
        
        with gr.Tabs():
            # Tab per messaggio singolo
            with gr.TabItem("📝 Messaggio Singolo"):
                gr.Markdown("### Analizza un singolo messaggio")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Messaggio da analizzare",
                            placeholder="Inserisci qui il messaggio...",
                            lines=5
                        )
                        analyze_btn = gr.Button("🔍 Analizza", variant="primary", size="lg")
                        
                        # Esempi
                        gr.Examples(
                            examples=[
                                ["Ciao, come stai? Bella giornata oggi!"],
                                ["Sei proprio stupido e non capisci niente!"],
                                ["Questo streaming è fantastico, complimenti!"],
                                ["Basta con questa roba, siete tutti dei deficienti"],
                                ["Grazie per la bella serata, ci sentiamo domani"]
                            ],
                            inputs=text_input,
                            label="Esempi di messaggi"
                        )
                    
                    with gr.Column(scale=1):
                        text_output = gr.Markdown(label="Risultati Analisi")
                
                analyze_btn.click(
                    fn=analyze_single_text,
                    inputs=text_input,
                    outputs=text_output
                )
            
            # Tab per file CSV
            with gr.TabItem("📈 Analisi CSV Chat"):
                gr.Markdown("### Analizza un file CSV di chat completo")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **Formato CSV richiesto:**
                        ```
                        author,message,badges,emotes
                        alexanderdc95,il re e il principe,['premium'],*[]*
                        brighi05,ma eri da sara secondi fa,['premium'],*[]*
                        sono_tuo_nonno22,il re,[],*[]*
                        ```
                        
                        **Colonne obbligatorie:**
                        - `author`: Nome utente
                        - `message`: Testo del messaggio
                        - `badges`: Lista badges (es: ['premium', 'vip'])
                        - `emotes`: Lista emotes (es: *['smile', 'heart']*)
                        """)
                        
                        file_input = gr.File(
                            label="Carica file CSV",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        analyze_csv_btn = gr.Button("📊 Analizza CSV", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        file_output = gr.Markdown(label="Risultati Analisi Chat")
                
                # Sezione grafici
                gr.Markdown("## 📊 Grafici Riassuntivi")
                
                with gr.Row():
                    with gr.Column():
                        chart1 = gr.Plot(label="Distribuzione Tossicità")
                    with gr.Column():
                        chart2 = gr.Plot(label="Top Utenti Tossici")
                
                with gr.Row():
                    with gr.Column():
                        chart3 = gr.Plot(label="Distribuzione Punteggi")
                    with gr.Column():
                        gr.Markdown("""
                        ### 📋 Legenda Grafici
                        
                        **Distribuzione Tossicità:**
                        - 🟢 Non Tossico (0-30%)
                        - 🟡 Moderatamente Tossico (30-70%)
                        - 🔴 Altamente Tossico (70-100%)
                        
                        **Top Utenti Tossici:**
                        - Mostra i 10 utenti con tossicità media più alta
                        - Include badges e emotes utilizzate
                        
                        **Distribuzione Punteggi:**
                        - Istogramma di tutti i punteggi di tossicità
                        - Aiuta a capire la distribuzione generale
                        """)
                
                analyze_csv_btn.click(
                    fn=analyze_csv_file_interface,
                    inputs=file_input,
                    outputs=[file_output, chart1, chart2, chart3]
                )
        
        gr.Markdown("""
        ---
        ### ℹ️ Informazioni Tecniche
        
        **Funzionalità:**
        - ✅ Analisi singoli messaggi con classificazione automatica
        - ✅ Analisi batch di file CSV con migliaia di messaggi
        - ✅ Visualizzazione badges e emotes per utente
        - ✅ Grafici interattivi per statistiche generali
        - ✅ Top utenti e messaggi più tossici
        - ✅ Statistiche dettagliate per utente
        
        **Note:**
        - Il modello restituisce una probabilità di tossicità da 0% a 100%
        - I risultati sono basati sul modello addestrato e potrebbero necessitare di calibrazione
        - Per file CSV di grandi dimensioni, l'analisi potrebbe richiedere alcuni minuti
        - I grafici sono interattivi e permettono zoom e selezione
        """)
    
    return interface

if __name__ == "__main__":
    
    interface = create_interface()
    interface.launch(
        share=True,  
        server_name="0.0.0.0",  
        server_port=7860  
    )