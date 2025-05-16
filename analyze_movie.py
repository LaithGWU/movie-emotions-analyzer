import glob
import os
import cv2
from deepface import DeepFace
import requests
import argparse
import shutil
import tensorflow as tf
import ffmpeg

TMDB_API_KEY = "ENTER YOUR OWN"
ACTOR_PROFILE_DIR = "actor_profiles"
FRAME_OUTPUT_DIR = "extracted_frames"
MATCHED_FACES_DIR = "matched_faces"    
FRAME_OUTPUT_DIR = "extracted_frames"                     

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) 
    os.makedirs(folder_path)

def extract_frames_to_dir(input_video):
    reset_folder(FRAME_OUTPUT_DIR)
    (
        ffmpeg  
        .input(input_video)
        .filter("fps", fps=1)
        .output(f"{FRAME_OUTPUT_DIR}/frame_%04d.jpg")
        .run()
    )  

def identify_actor(frame_path, threshold=0.6):
    for filename in os.listdir(ACTOR_PROFILE_DIR):

        actor_id = os.path.splitext(filename)[0]
        actor_image_path = os.path.join(ACTOR_PROFILE_DIR, filename)

        try:
            result = DeepFace.verify(img1_path=frame_path, img2_path=actor_image_path, enforce_detection=False)

            if result["verified"] and result["distance"] < threshold:
                return actor_id
        except Exception as e:
            print(f"Error comparing with {filename}: {e}")

    return -1

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
        print(f"{title} ({release_date}) - {overview}")
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

    profile_path = profiles[0].get("file_path")
    if not profile_path:
        print(f"No file path found in the first profile")
        return

    image_url = f"https://image.tmdb.org/t/p/w500/{profile_path}"

    image_data = requests.get(image_url).content

    file_path = os.path.join(ACTOR_PROFILE_DIR, f"{person_id}.jpg")
    
    with open(file_path, "wb") as f:
        f.write(image_data)

def face_verified(face_bgr, actor_image_path):
    """
    Return (True, distance) if the face crop matches the actor image.
    face_bgr : numpy array from cv2 with shape (h,w,3) in BGR
    """
    if face_bgr is None or face_bgr.size == 0:
        return False, None           # empty crop

    # DeepFace expects RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    result = DeepFace.verify(
        img1_path=face_rgb,                  # numpy array instead of file path
        img2_path=actor_image_path,          # existing jpg on disk
        enforce_detection=False,             # don’t crash if face detector sees nothing
        detector_backend="skip"              # skip detector because we already cropped
    )

    return result["verified"]


def detect_and_match_faces(frame_path):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"Could not load {frame_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for i, (x, y, w, h) in enumerate(faces):
        print("faces detected")

        crop = frame[y:y + h, x:x + w]

        for fname in os.listdir(ACTOR_PROFILE_DIR):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            actor_id = os.path.splitext(fname)[0]
            actor_img = os.path.join(ACTOR_PROFILE_DIR, fname)

            try:
                ok = face_verified(crop, actor_img)
                if ok:
                    save_to = os.path.join(
                        MATCHED_FACES_DIR,
                        f"{actor_id}_from_{os.path.basename(frame_path)}_face{i}.jpg"
                    )
                    cv2.imwrite(save_to, crop)
                    print(f"✔  {actor_id}  →  {save_to}")
                    break               # stop checking other actors
            except Exception as e:
                print(f"⚠  {fname}: {e}")

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    parser = argparse.ArgumentParser(description="A movie emotion analyzer.")
    parser.add_argument("file", type=str, help="File to analyze")
    parser.add_argument("title", type=str, help="Movie title to search for metadata")

    args = parser.parse_args()

    movie_id = get_movie_id(args.title)
    cast = get_actors(movie_id)

    reset_folder(ACTOR_PROFILE_DIR)

    for actor in cast[:5]:
        print(f"{actor['id']} {actor['name']} as {actor['character']}")
        get_actor_profile(actor['id'])

    extract_frames_to_dir(args.file)

    frame_files = sorted(glob.glob(f"{FRAME_OUTPUT_DIR}/*.jpg"))

    results = []

    reset_folder(MATCHED_FACES_DIR)

    for frame_path in frame_files:
        detect_and_match_faces(frame_path)