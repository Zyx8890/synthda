from video_processing.motioncaptioning import process_videos
from video_processing.models import action_class
import subprocess
from pathlib import Path
import shutil
import os
from optimisation.optimisation_real_synth import main_synth_real
from dotenv import dotenv_values
from utils.blender_utils import npy_to_video, find_file_by_weights


env_vars = dotenv_values(".env")  

# ==============================
# CONFIGURATION - Define Key Paths
# ==============================

StridedTransformer_path = env_vars.get("STRIDED_TRANSFORMER")
text_to_motion_path = env_vars.get("TEXT_TO_MOTION")

def auto_npy_generation(video_files, video_name, StridedTransformer_path, text_to_motion_path):

    """
    Automates the process of generating .npy motion files from a video.
    
    Steps:
    1. Extracts motion description using ChatGPT.
    2. Generates text-based motion using the text-to-motion repository.
    3. Processes motion through StridedTransformer-Pose3D.
    4. Optimizes and combines motion data.
    """

    # Extract the folder name from the video file (e.g., "07" from "07.mp4")
    video_folder_name = video_name.replace(".mp4", "")

    # Define the output folder where results will be stored
    folder_path = f"{str(output_directory)}/{video_folder_name}/"

    print(f"folder path name is called: {folder_path} ")
    
    os.makedirs(folder_path, exist_ok=True)  # ✅ Ensure the folder exists
    
    # checking if real_path_npz exists, if it does, skip to genereation of video
    npz_found = folder_path + "output_keypoints_3d.npz"

    if not os.path.exists(npz_found):

        # ==============================
        # 1️⃣ EXTRACT VIDEO DESCRIPTION (ChatGPT)
        # ==============================

        model_name = "ChatGPT"  # Model used for video description
        api_key = env_vars.get("GPT_APIKEY") # OpenAI API Key
        additional_info = ["Describe the person in the video"]  # Instruction for ChatGPT

        # try statemtent in case video has error
        try:
            # Generate textual description of the video
            results = process_videos(video_files, model_name, api_key, *additional_info)
            
        except Exception as e:
            print(f"Error processing videos: {e}")
            
            # default description if the generated text from chatgpt doesntwork, can be keyed in videos_processing/models.py file
            results = action_class
            
        print(f"Log: Response from ChatGPT - {results}")

        # Save the generated text description into a text file
        text_file_path = os.path.join(folder_path, "input.txt")
        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write(results) 

        print(f"Log: Saving response as a text file and copying to text-to-motion repo")

        # ==============================
        # 2️⃣ GENERATE SYNTHETIC-MOTION DATA (Text-to-Motion)
        # ==============================

        # Copy the generated input text to the text-to-motion repository
        destination = text_to_motion_path + "/input.txt"
        shutil.copy(text_file_path, destination)  

        # Define and execute the command for motion generation
        text_to_motion_path = Path(text_to_motion_path)

        print("Log: Running text-to-motion repo")

        command = [
            "python", "gen_motion_script.py",
            "--name", "Comp_v6_KLD01",
            "--text_file", "input.txt",
            "--repeat_time", "1",
            "--result_path", f"{str(output_directory.resolve())}/{video_folder_name}/"
        ]

        # Run the script inside the text-to-motion repo
        subprocess.run(command, cwd=text_to_motion_path)

        print("Log: text-to-motion repo completed, retrieving synthetic_path")

        # Finding the generation .npy file from text-to-motion
        synthetic_folder_path = folder_path + "t2m/Comp_v6_KLD01/default/animations/C000/"
        
        dir_synthetic_folder_path = Path(synthetic_folder_path)
        npy_files = sorted(dir_synthetic_folder_path.glob("*.npy"))

        if not npy_files:
            raise FileNotFoundError(f"🚨 ERROR: No .npy file found in {synthetic_folder_path}")

        synthetic_path_name = npy_files[-1].name if npy_files else None
        
        # important path to feed into main_synth_real_function
        synthetic_path = synthetic_folder_path + synthetic_path_name

        # copying that file to the current folder
        final_synthetic_path = folder_path + synthetic_path_name
        shutil.copy(synthetic_path, final_synthetic_path)

        # ==============================
        # 3️⃣ GENERATE TRACKED-MOTION DATA (Strided Transformer) 
        # ==============================
        
        # Define the path to StridedTransformer-Pose3D
        StridedTransformer_path = Path(StridedTransformer_path)

        # Copy the video file to StridedTransformer for processing
        destination = StridedTransformer_path / "demo/video" / video_name
        shutil.copy(video_files[0], destination)

        print("Log: Video copied to StridedTransformer, running it!")

        # Run the StridedTransformer script for 3D pose estimation
        subprocess.run(["python", "demo/vis.py", "--video", video_name], cwd=StridedTransformer_path)

        print("Log: StridedTransformer processing completed")

        # Retrieving generation .npz file from StridedTransformer Repo
        copying_from = StridedTransformer_path / "demo/output" / video_folder_name / "output_3D/output_keypoints_3d.npz"
        real_path_npz = folder_path + 'output_keypoints_3d.npz'

        if not copying_from.exists():
            raise FileNotFoundError(f"🚨 ERROR: Generated NPZ file not found in {copying_from}")
        
        shutil.copy(copying_from, real_path_npz)

        print("Log: Copied generated NPZ file to dataset folder")

        # ==============================
        # 4️⃣  OPTIMIZE MOTION DATA
        # ==============================

        print("Log: Running optimization with real and synthetic motion data")

        main_synth_real(real_path_npz, final_synthetic_path, folder_path)

        print("🎉 Motion optimization completed!")

    else:
        print("Log: Skipping generation of synthetic data .npy files")

    # ==============================
    # 5️⃣ Generating the video
    # ==============================

    # all_variations_folder_path name
    variation_folder = folder_path + "all_variations"

    weight_B = round(1-weight_A,2)
    weights = (weight_A, weight_B)

    npy_file_path = find_file_by_weights(variation_folder, weights)
    
    generated_video_path = npy_to_video(video_folder_name, npy_file_path)

    source_video = Path(generated_video_path)
    destination_video = video_generated_path / source_video.name
    shutil.copy(source_video, destination_video)

    print(f"VIDEO GENERATING NOW!!!")


# ==============================
# Main function to be used
# ==============================

def syn_real_main(weight_A_value, input_directory_path, output_directory_path):

    global weight_A, video_generated_path, videos_path, video_directory, output_directory

    # Directory containing the MP4 video files to process (Rmbr to change)
    videos_path = input_directory_path

    video_directory = Path(videos_path)  

    # Folder where the original videos will be processed and the folders of each video will be created
    output_directory = Path(output_directory_path)

    weight_A = weight_A_value # use the passed value from the function

    # creating a folder called videos_generated
    # Define original and generated video paths
    video_generated_path = video_directory.parent / f"videos_generated_{weight_A}"  # Replace "videos" with "videos_generated"

    # Create the folder if it doesn't exist
    video_generated_path.mkdir(parents=True, exist_ok=True)

    # ==============================
    # PROCESS MULTIPLE VIDEOS
    # ==============================
    
    # Get all MP4 files in the video directory
    video_files = sorted(video_directory.glob("*.mp4"))  # Retrieve a list of all MP4 video paths
    
    print(f"Videos found in folder are {video_files}")

    # Loop through each MP4 file and process it individually
    for video_path in video_files:
        video_name = video_path.stem  # Extract filename without extension (e.g., "07" from "07.mp4")

        # Create a dedicated folder for each video in the output directory
        video_folder = output_directory / video_name
        video_folder.mkdir(parents=True, exist_ok=True)  # ✅ Ensure the folder exists

        print(f"📂 Created folder: {video_folder}")

        # Copy the video file into its dedicated folder
        destination_video_path = video_folder / video_path.name
        shutil.copy(video_path, destination_video_path)  # ✅ Copy the video to its processing folder

        print(f"🎥 Copied video to: {destination_video_path}")

        # Convert video name back to its original MP4 format for processing
        video_name = video_name + ".mp4"

        # Update video_files list to contain only the copied video
        video_files = [destination_video_path]

        # Start the full processing pipeline for this video
        auto_npy_generation(video_files, video_name, StridedTransformer_path, text_to_motion_path)
