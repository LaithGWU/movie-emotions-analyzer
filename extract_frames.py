import ffmpeg   
import os
import shutil


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