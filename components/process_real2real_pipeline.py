import subprocess
from pathlib import Path
import shutil
import os
from optimisation.optimisation_both_real import main_real_real
from dotenv import dotenv_values
from itertools import combinations
import random
from utils.blender_utils import npy_to_video, find_file_by_weights


env_vars = dotenv_values(".env")  

# ==============================
# CONFIGURATION - Define Key Paths
# ==============================

StridedTransformer_path = env_vars.get("STRIDED_TRANSFORMER")

def auto_npy_generation(video_files_1, video_files_2, video_folder_name, StridedTransformer_path):

    # video_name
    video_name_1 = Path(video_files_1).name.replace(".mp4","")
    video_name_2 = Path(video_files_2).name.replace(".mp4","")

    """
    Automates the process of generating .npy motion files from a video.
    
    Steps:
    1. Extracts motion description using ChatGPT.
    2. Processes motion through StridedTransformer-Pose3D for both real videos.
    3. Optimizes and combines motion data.
    """

    # Define the output folder where results will be stored
    folder_path = str(output_directory / video_folder_name)

    print(f"folder path name is called: {folder_path} ")

    os.makedirs(folder_path, exist_ok=True)  # ✅ Ensure the folder exists

    # checking if real_path_npz of video_1 exists, if it does, skip to genereation of video
    npz_found = folder_path + "output_keypoints_3d" + "_" + video_name_1 + ".npz"    

    if not os.path.exists(npz_found):

        # ==============================
        # 1️⃣ GENERATE TRACKED-MOTION DATA FOR BOTH VIDEOS (Strided Transformer) 
        # ==============================

        # Define the path to StridedTransformer-Pose3D
        StridedTransformer_path = Path(StridedTransformer_path)

        # Copy the video files to StridedTransformer for processing
        destination_1 = StridedTransformer_path / "demo/video" / video_name_1
        destination_2 = StridedTransformer_path / "demo/video" / video_name_2

        shutil.copy(video_files_1, destination_1)
        shutil.copy(video_files_2, destination_2)

        print("Log: Video copied to StridedTransformer, running it!")

        # Run the StridedTransformer script for 3D pose estimation
        subprocess.run(["python", "demo/vis.py", "--video", video_name_1], cwd=StridedTransformer_path)
        subprocess.run(["python", "demo/vis.py", "--video", video_name_2], cwd=StridedTransformer_path)

        print("Log: StridedTransformer processing completed")
        
        # Retrieving generation .npz files from StridedTransformer Repo
        real_path_npz_1 = folder_path + '/output_keypoints_3d' + "_" + video_name_1 + '.npz'
        real_path_npz_2 = folder_path + '/output_keypoints_3d' + "_" + video_name_2 + '.npz'

        copying_from_1 = StridedTransformer_path / "demo/output" / video_name_1 / "output_3D/output_keypoints_3d.npz"

        copying_from_2 = StridedTransformer_path / "demo/output" / video_name_2 / "output_3D/output_keypoints_3d.npz"

        if not copying_from_1.exists():
            raise FileNotFoundError(f"🚨 ERROR: Generated NPZ file not found in {copying_from_1}")

        if not copying_from_2.exists():
            raise FileNotFoundError(f"🚨 ERROR: Generated NPZ file not found in {copying_from_2}")

        shutil.copy(copying_from_1, real_path_npz_1)
        shutil.copy(copying_from_2, real_path_npz_2)

        print("Log: Copied generated NPZ file to dataset folder")

        # ==============================
        # 2️⃣ OPTIMIZE MOTION DATA
        # ==============================

        print("Log: Running optimization with real and synthetic motion data")

        print(f"variable {folder_path}")

        main_real_real(real_path_npz_1, real_path_npz_2, folder_path)

        print("🎉 Motion optimization completed!")

    else:
        print("Log: Skipping generation of synthetic data .npy files")

    # ==============================
    # 3️⃣ Generating the video
    # ==============================

    # all_variations_folder_path name
    variation_folder = folder_path + "/all_variations"

    weight_B = round(1-weight_A,2)
    weights = (weight_A, weight_B)

    npy_file_path = find_file_by_weights(variation_folder, weights)
    
    generated_video_path = npy_to_video(video_folder_name, npy_file_path)

    source_video = Path(generated_video_path)
    destination_video = video_generated_path / source_video.name
    shutil.copy(source_video, destination_video)

    print(f"VIDEO GENERATING NOW!!!")

def both_real_main(weight_A_value, input_directory_path, output_directory_path, number_of_videos):

    global weight_A, video_generated_path, videos_path, video_directory, output_directory

    # Directory containing the MP4 video files to process
    videos_path = input_directory_path
    video_directory = Path(videos_path)  

    # Root dataset directory where results and extracted data will be saved
    output_directory = Path(output_directory_path)  

    # random seed
    random.seed(42)

    # number of videos desired
    number_of_videos_desired = number_of_videos

    # Weights declaration (1.dp)
    weight_A = weight_A_value

    # creating a folder called videos_generated
    # Define original and generated video paths
    video_generated_path = video_directory.parent / f"videos_generated_real2_{weight_A}"  # Replace "videos" with "videos_generated"

    # Create the folder if it doesn't exist
    video_generated_path.mkdir(parents=True, exist_ok=True)

    # ==============================
    # PROCESS MULTIPLE VIDEOS
    # ==============================

    # Get all MP4 files in the video directory
    mp4_videos = list(video_directory.glob("*.mp4"))
    mp4_videos = [str(video) for video in mp4_videos]

    print(f"Found {len(mp4_videos)} videos in {video_directory}")

    # Ensure the desired number of videos is possible
    comb_list = list(combinations(mp4_videos, 2))
        
    # Ensure the desired number of videos is possible
    if number_of_videos_desired <= len(comb_list):
        random_selection = random.sample(comb_list, number_of_videos_desired)
    else:
        print("⚠️ Not enough combinations available! Using all available combinations.")
        random_selection = []

    for video_1_path, video_2_path in random_selection:

        # Path type
        video_1_path_type = Path(video_1_path)
        video_2_path_type = Path(video_2_path)

        name_1 = video_1_path_type.stem
        name_2 = video_2_path_type.stem

        # folder_title
        video_name = name_1 + "_" + name_2
        
        # Create a dedicated folder for each video in the output directory
        video_folder = output_directory / video_name
        video_folder.mkdir(parents=True, exist_ok=True) # ✅ Ensure the folder exists
        print(f"📂 Created folder: {video_folder}")

        # Copy the video files into its dedicated folder
        destination_video_path_1 = video_folder / video_1_path_type.name
        destination_video_path_2 = video_folder / video_2_path_type.name

        shutil.copy(video_1_path_type, destination_video_path_1) 
        shutil.copy(video_2_path_type, destination_video_path_2) 

        print(f"🎥 Copied video to: {destination_video_path_1} & {destination_video_path_1} ")

        # Start the full processing pipeline for this video
        auto_npy_generation(video_1_path, video_2_path, video_name, StridedTransformer_path)
