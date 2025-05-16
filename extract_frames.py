import ffmpeg   
import os
import shutil

FRAME_OUTPUT_DIR = "extracted_frames"
MOVIE_FILE = "the_shining.mkv"

def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) 
    os.makedirs(folder_path)  


reset_folder(FRAME_OUTPUT_DIR)
(
    ffmpeg  
    .input(f"./movies/{MOVIE_FILE}")
    .filter("fps", fps=1)
    .output(f"{FRAME_OUTPUT_DIR}/frame_%04d.jpg")
    .run()
)