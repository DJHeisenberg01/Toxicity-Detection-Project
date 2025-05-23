import json
import pandas as pd
import os
from typing import List, Dict, Optional
from chat_downloader import ChatDownloader
from chat_downloader.sites import TwitchChatDownloader
import time
from datetime import datetime



class TwitchDataCollector:
    
    def __init__(self):
        self.chat_downloader = ChatDownloader()
        self.twitch_downloader = TwitchChatDownloader()
        
    def get_user_vods(self, username: str, limit: int = 5, sort_by: str = "TIME") -> List[Dict]:
        
        print(f" Recuperando VOD per {username}...")
        vods = self.twitch_downloader.get_vods(username, limit=limit, sort_by=sort_by)
        
        vod_list = []
        for vod in vods:
            vod_list.append(vod)
        
        return vod_list
    
    def extract_chat_from_vod(self, vod_id: str) -> List[Dict]:
        
        vod_url = f"https://www.twitch.tv/videos/{vod_id}"
        print(f" Recuperando chat per {vod_url}...")
        
        start_time = time.time()
        chat = self.chat_downloader.get_chat(vod_url)
        
        messages = []
        for message in chat:
            if message.get('message_type') == 'text_message':
                author = message.get("author", {})
                name = author.get("name", "")
                message_text = message.get("message", "")
                badges_full = author.get("badges", [])
                
                # Estrai solo i nomi dei badge
                badge_names = [badge.get("name", "") for badge in badges_full]
                
                messages.append({
                    "author": name,
                    "message": message_text,
                    "badges": badge_names,
                    "timestamp": message.get("timestamp", 0),
                    "time_in_seconds": message.get("time_in_seconds", 0)
                })
        elapsed_time = time.time() - start_time
        print(f" Chat recuperato in {elapsed_time:.2f} secondi")
        
        return messages
    
    def collect_chats_from_user(self, username: str, n_vods: int = 3) -> Dict[str, List[Dict]]:
        
        print(f"Raccolta chat per {username}...")
        
        vods = self.get_user_vods(username, limit=n_vods)
        
        vod_ids = [vod['id'] for vod in vods]
        print(f"📋 VOD IDs: {vod_ids}")
        
        all_chats = []
        for i, vod_id in enumerate(vod_ids, 1):
            print(f"📥 Processando VOD {i}/{len(vod_ids)}: {vod_id}")
            vod_url = f"https://www.twitch.tv/videos/{vod_id}"
            
            try:
                messages = self.extract_chat_from_vod(vod_id)
                all_chats[vod_url] = messages
            except Exception as e:
                print(f"❌ Errore nel processare VOD {vod_id}: {e}")
                continue
            
        print(f"Raccolta completa! Ottenuti {len(all_chats)} VOD con chat")
        return all_chats
    
    def save_chats_to_json(self, chats_data: Dict, filename: str):
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(chats_data, f, indent=4, ensure_ascii=False)
        
        print(f"Dati salvati in {filename}")



