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

    # Di ogni emote, prende solo il nome
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
    num_streamers = range(1, len(streamer_names))
    
    for i, streamer in zip(num_streamers, streamer_names):
        
        emote_names_list = []
        print(f"{i}) Recuperando le emotes di: {streamer}")
        
        # Informazioni sullo streamer
        streamer_info = await first(twitch.get_users(logins=streamer))

        # Get delle emotes dello streamer a partire dal suo ID
        streamer_emotes = await twitch.get_channel_emotes(streamer_info.id)
        
        # Di ogni emote, trattieni solo il nome
        for emote in streamer_emotes.data:
            emote_names_list.append(emote.name)
        
        # Aggiungi le emotes dello streamer al dizionario (se ne ha)
        if emote_names_list:
            emotes_dict[streamer] = emote_names_list

    return emotes_dict



import aiohttp

async def fetch_json(session, url):
    async with session.get(url) as response:
        return await response.json()

async def get_bttv_emotes(user_id, session):
    emotes = []
    global_data = await fetch_json(session, "https://api.betterttv.net/3/cached/emotes/global")
    emotes.extend([emote["code"] for emote in global_data])
    try:
        user_data = await fetch_json(session, f"https://api.betterttv.net/3/cached/users/twitch/{user_id}")
        for category in ("channelEmotes", "sharedEmotes"):
            emotes.extend([emote["code"] for emote in user_data.get(category, [])])
    except:
        pass
    return emotes

async def get_ffz_emotes(streamer_name, session):
    emotes = []
    try:
        global_data = await fetch_json(session, "https://api.frankerfacez.com/v1/set/global")
        for set_data in global_data["sets"].values():
            emotes.extend([emote["name"] for emote in set_data["emoticons"]])
        room_data = await fetch_json(session, f"https://api.frankerfacez.com/v1/room/{streamer_name}")
        set_id = room_data["room"]["set"]
        set_data = room_data["sets"][str(set_id)]
        emotes.extend([emote["name"] for emote in set_data["emoticons"]])
    except:
        pass
    return emotes

async def get_7tv_emotes(user_id, session):
    emotes = []
    try:
        global_data = await fetch_json(session, "https://api.7tv.app/v2/emotes/global")
        emotes.extend([emote["name"] for emote in global_data])
        user_data = await fetch_json(session, f"https://api.7tv.app/v2/users/{user_id}/emotes")
        emotes.extend([emote["name"] for emote in user_data])
    except:
        pass
    return emotes
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

    async with aiohttp.ClientSession() as session:
        for streamer in streamer_names:
            streamer_info = await first(twitch.get_users(logins=streamer))
            user_id = streamer_info.id

            bttv_emotes = await get_bttv_emotes(user_id, session)
            ffz_emotes = await get_ffz_emotes(streamer, session)
            seventv_emotes = await get_7tv_emotes(user_id, session)

            if bttv_emotes:
                all_emotes[streamer + "_BTTV"] = bttv_emotes
            if ffz_emotes:
                all_emotes[streamer + "_FFZ"] = ffz_emotes
            if seventv_emotes:
                all_emotes[streamer + "_7TV"] = seventv_emotes

    # Salvataggio delle emotes in un file JSON
    with open("api/all_emotes.json", "w") as f:
        json.dump(all_emotes, f, indent=4)

    print("Emotes raccolte con successo! La connessione all'API verrà chiusa.")

    # Chiusura della connessione all'API
    await twitch.close()


if __name__ == "__main__":
    # Lista di streamer da cui ottenere le emotes
    streamer_names = []
    
    with open("api/top200streamers.txt", "r") as file:
        for line in file:
            # Aggiungi ogni nome di streamer alla lista
            streamer_names.append(line.strip())
    
    # Avvio della funzione principale
    asyncio.run(main(streamer_names))


