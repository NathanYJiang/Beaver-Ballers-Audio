#Task 1 
import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from typing import Union, Callable, Tuple
from pathlib import Path
from microphone import record_audio

def record_and_save():
    listen_time = 10 # seconds
    frames, sample_rate = record_audio(listen_time)

    return (frames, sample_rate)

frames, sample_rate = record_and_save()

samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
array_to_save = np.hstack((sample_rate, samples))

np.save('/Users/manitmehta/Desktop/Cogworks', array_to_save)

def load_and_parse(file_path: str) -> Tuple[np.ndarray, int]:
    loaded_array = np.load(file_path)
    
    sample_rate = loaded_array[0]
    signal = loaded_array[1:]
    
    return signal, sample_rate
    
songs_db = {}
song_id = 1

def add_song(title, artist, file_path):
    global song_id
    songs_db[song_id] = {
        'title': title,
        'artist': artist,
        'file_path': file_path
    }
    song_id += 1

def get_song(id):
    print(songs_db.get(id))
    return songs_db.get(id)
    
def get_all_songs():
    print(list(songs_db.values()))
    return list(songs_db.values())
    
def get_all_artists():
    print(list({song['artist'] for song in songs_db.values()}))
    return list({song['artist'] for song in songs_db.values()})

def get_all_song_titles():
    print([song['title'] for song in songs_db.values()])
    return [song['title'] for song in songs_db.values()]



    





#listen_time = 10 #audio seconds


print('end')





