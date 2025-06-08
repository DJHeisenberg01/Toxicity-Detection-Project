import asyncio
import json
from twitchAPI.twitch import Twitch
from twitchAPI.helper import first
from api_keys import CLIENT_ID, SECRET_ID

APP_ID = CLIENT_ID
APP_SECRET = SECRET_ID


async def get_all_emotes(twitch):
    emotes_dict = {}
    emote_names_list = []

    # Get delle emotes globali
    global_emotes = await twitch.get_global_emotes()

    # Di ogni emote, trattieni solo il nome
    for emote in global_emotes.data:
        emote_names_list.append(emote.name)
    
    # Aggiungi le emotes globali al dizionario
    emotes_dict["global"] = emote_names_list
    
    return emotes_dict


async def get_streamers_emotes(twitch, streamer_names):
    emotes_dict = {}
    
    # Connessione all'API di Twitch
    twitch = await Twitch(APP_ID, APP_SECRET)

    # Itera sulla lista di stramer da cui prendere le emotes
    for streamer in streamer_names:
        emote_names_list = []
        
        # Informazioni sullo streamer
        streamer_info = await first(twitch.get_users(logins=streamer))

        # Get delle emotes dello streamer a partire dal suo ID
        streamer_emotes = await twitch.get_channel_emotes(streamer_info.id)
        
        # Di ogni emote, trattieni solo il nome
        for emote in streamer_emotes.data:
            emote_names_list.append(emote.name)
        
        # Aggiungi le emotes dello streamer al dizionario
        emotes_dict[streamer] = emote_names_list

    return emotes_dict


async def main(streamer_names):
    # Connessione all'API di Twitch
    twitch = await Twitch(APP_ID, APP_SECRET)

    # Get delle emotes globali e degli streamer specificati
    global_emotes = await get_all_emotes(twitch)
    streamer_emotes = await get_streamers_emotes(twitch, streamer_names)

    # Unione di tutte le emotes in un unico dizionario
    all_emotes = {}
    all_emotes.update(global_emotes)
    all_emotes.update(streamer_emotes)

    # Salvataggio delle emotes in un file JSON
    with open("api/all_emotes.json", "w") as f:
        json.dump(all_emotes, f, indent=4)

    print("Emotes raccolte con successo! La connessione all'API verrà chiusa.")

    # Chiusura della connessione all'API
    await twitch.close()


if __name__ == "__main__":
    # Lista di streamer da cui ottenere le emotes
    streamer_names = ["lollolacustre", "dariomocciatwitch"]
    
    # Avvio della funzione principale
    asyncio.run(main(streamer_names))


