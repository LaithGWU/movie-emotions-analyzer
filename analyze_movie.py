import glob
import os
import cv2
from deepface import DeepFace
import requests
import argparse
import shutil
import tensorflow as tf
from collections import defaultdict, Counter 
import csv
import uuid
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TMDB_API_KEY = "ENTER YOUR OWN"

ACTOR_PROFILE_DIR = "actor_profiles"
FRAME_OUTPUT_DIR = "extracted_frames"
MATCHED_FACES_DIR = "matched_faces"
GRAPH_OUTPUT_DIR = "piecharts"

MODEL      = "Facenet512"
DETECTOR   = "retinaface"

emotion_tally = defaultdict(Counter)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)

def get_movie_id(title):
    url = f"https://api.themoviedb.org/3/search/movie?query={title}&include_adult=false&language=en-US&page=1"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    for movie in data["results"]:
        title = movie.get("title")
        release_date = movie.get("release_date")
        overview = movie.get("overview")

        print(f"{title} ({release_date}) - \n---------------------------------------------------------------------\n{overview}\n---------------------------------------------------------------------")

        user_input = input("Is this the movie you wanted (y/n)?: ")

        if user_input.startswith("y"):
            return movie.get("id")
    
    print("Could not find the movie you requested")

    exit(0)


def get_actors(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{str(movie_id)}/credits?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    cast = []

    for person in data.get("cast", []):
        if person.get("known_for_department") == "Acting":
            cast.append({"id": person["id"], "name": person["name"], "character": person["character"]})
           
    return cast

def get_actor_profile(person_id):
    url = f"https://api.themoviedb.org/3/person/{person_id}/images"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    profiles = data.get("profiles", [])

    if not profiles:
        print(f"No profile images found for person ID {person_id}")
        return
    
    reset_folder(os.path.join(ACTOR_PROFILE_DIR, str(person_id)))

    for i, profile in enumerate(profiles):
        profile_path = profile.get("file_path")

        if not profile_path:
            print(f"No file path found in the first profile")
            return

        image_url = f"https://image.tmdb.org/t/p/w500/{profile_path}"

        image_data = requests.get(image_url).content

        file_path = os.path.join(ACTOR_PROFILE_DIR, str(person_id), f"{i}.jpg")
        
        with open(file_path, "wb") as f:
            f.write(image_data)

def initialize_database():
    dummy = glob.glob(f"{ACTOR_PROFILE_DIR}/*/*.jpg")
    DeepFace.find(
        img_path=str(dummy[0]),
        db_path=ACTOR_PROFILE_DIR,
        model_name=MODEL,
        detector_backend=DETECTOR,
        silent=True,
        enforce_detection=False,
        distance_metric="cosine"
    )

def get_emotion(face, actor_id):
    try:
        analysis = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            align=True
        )
        
        dominant = analysis[0]["dominant_emotion"]

        if dominant in EMOTION_LABELS:
            emotion_tally[actor_id][dominant] += 1
            print(f"{actor_id} | emotion: {dominant}")
            return dominant

    except Exception as e:
        print(f"emotion analysis failed: {e}")


def analyze_frame(frame_path):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return

    results = DeepFace.find(
    img_path=frame,                    
    db_path=ACTOR_PROFILE_DIR,
    model_name=MODEL,
    detector_backend=DETECTOR,    
    distance_metric="cosine",
    enforce_detection=False,
    silent=True,
    align=True,
    threshold=0.55
    )

    for face in results:
        if face.empty:
            continue
            
        identity = face.iloc[0]["identity"]
        actor_id = Path(identity).parts[-2] 

        x,y,w,h = map(int, [face.iloc[0]["source_x"], face.iloc[0]["source_y"],
                            face.iloc[0]["source_w"], face.iloc[0]["source_h"]])
        
        crop = frame[y:y+h, x:x+w]

        emotion = get_emotion(crop, actor_id)
        if emotion == None:
            emotion = "unknown"

        uid = uuid.uuid4().hex
        save_to = os.path.join(MATCHED_FACES_DIR, actor_id, emotion)

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        save_to =  os.path.join(save_to, f"{uid}.jpg")

        cv2.imwrite(save_to, crop)

def save_emotion_csv(filename="actor_emotions.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actor_id", "emotion", "count"])

        for actor_id, counter in emotion_tally.items():
            for emotion in EMOTION_LABELS:
                writer.writerow([actor_id, emotion, counter.get(emotion, 0)])

def graph_emotions(movie, character_dict, filename="actor_emotions.csv"):
    df = pd.read_csv(filename)

    reset_folder(os.path.join(GRAPH_OUTPUT_DIR, str(movie)))

    for actor_id, sub in df.groupby("actor_id"):
        sub = sub[sub["count"] > 0]

        labels = sub["emotion"]
        sizes  = sub["count"]

        def autopct(pct):
            return f"{pct:.1f}%" if pct >= 5 else ""

        explode = [0.05 if s / sum(sizes) < 0.05 else 0 for s in sizes]

        plt.figure()
        plt.pie(
            sizes,
            labels=labels,
            autopct=autopct,
            explode=explode,
            pctdistance=0.8,     
            labeldistance=1.05 
        )
        plt.title(f"Emotion distribution – {character_dict[actor_id][0]} played by {character_dict[actor_id][1]}")
        plt.tight_layout()

        save_to = os.path.join(GRAPH_OUTPUT_DIR, str(movie), f"{actor_id}.png")
        plt.savefig(save_to, dpi=200)

        plt.close()
    

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    print(f"\n\n")

    parser = argparse.ArgumentParser(description="A movie emotion analyzer.")
    parser.add_argument("title", type=str, help="Movie title to search for metadata")
    args = parser.parse_args()

    movie_id = get_movie_id(args.title)

    print("getting actor profiles...")

    cast = get_actors(movie_id)
    character_dict = {}
    reset_folder(ACTOR_PROFILE_DIR)

    for actor in cast[:6]:
        print(f"{actor['id']} {actor['name']} as {actor['character']}")
        character_dict[actor['id']] = (actor['character'], actor['name'])
        get_actor_profile(actor['id'])

    print("initializing database...")
    initialize_database()

    frame_files = sorted(glob.glob(f"{FRAME_OUTPUT_DIR}/*.jpg"))

    reset_folder(MATCHED_FACES_DIR)

    print("analyzing frames...")
    for frame_path in frame_files:
        print(f"analyzing {frame_path}")
        analyze_frame(frame_path)

    save_emotion_csv()
    graph_emotions(movie_id, character_dict)