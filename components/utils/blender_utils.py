from pathlib import Path
import shutil
import os
import re
import subprocess
from dotenv import dotenv_values


env_vars = dotenv_values(".env")  

join2smpl_path = env_vars.get("JOIN2SMPL")
blender_path = env_vars.get("BLENDER")

# ==============================
# Converts npy to mp4 video
# ==============================

def npy_to_video(video_name, original_npy_file):

    # ==============================
    # 1️⃣ Convert .npy file to a folder of .obj files (joints2smpl)
    # ==============================

    # extracting the file name and duplicating to the joints2smpl folder
    filename = Path(original_npy_file).name
    join2smpl_npy_path = join2smpl_path + "/demo/demo_data/" + video_name + filename

    shutil.copy(original_npy_file, join2smpl_npy_path)

    # running the command for joints2smpl
    command = [
        "python", "fit_seq.py",
        "--files", video_name + filename,

        # changing the iterations here 
        "--num_smplify_iters", "1"
    ]

    # Run the script inside the text-to-motion repo
    subprocess.run(command, cwd=join2smpl_path)

    # blender scripts

    filename = Path(original_npy_file).name
    # duplicate the folder from output of the joints2smp to the blender test folder

    foldername = filename.replace(".npy","")

    join2smpl_output_path = join2smpl_path + "/demo/demo_results/" + video_name + foldername
    render_blender_path = blender_path + "/renders/" + video_name + foldername

    # Ensure the destination path exists, or deleting if it already exists
    if os.path.exists(render_blender_path):
        shutil.rmtree(render_blender_path)  # Remove existing folder

    shutil.copytree(join2smpl_output_path, render_blender_path)

    # ==============================
    # 2️⃣ Generate .mp4 video from .obj files (Blender)
    # ==============================

    # Running the command for blender (Blender 3.0.1)
    command_blender = [
        "./blender", "-b", "-P", f"{blender_path}/animation_pose.py", "--", "--name", video_name + foldername
    ]

    # Run the command
    subprocess.run(command_blender, cwd=blender_path)

    # Ensure test_blender_path is a Path object
    render_blender_path = Path(render_blender_path)

    # Now check for the .mp4 file
    mp4_files = sorted(render_blender_path.glob("*.mp4"))

    if not mp4_files:
        raise FileNotFoundError(f"🚨 ERROR: No .mp4 file found in {render_blender_path}.")

    mp4_path_name = mp4_files[-1].name  # Get the last sorted .mp4 file

    print(f"🎥 MP4 File Found: {mp4_path_name}")
   
    return render_blender_path / mp4_path_name 

# ==============================
# Finding the .npy file from the folder
# ==============================

def find_file_by_weights(folder_path, weights):
    # Regular expression to extract weights from the filename.
    pattern = r"euclidean_distances_wA([0-9.]+)_wB([0-9.]+)\.npy"
    
    # Iterate over files in the directory.
    for filename in os.listdir(folder_path):
        match = re.search(pattern, filename)
        if match:
            file_weight_A = float(match.group(1))
            file_weight_B = float(match.group(2))
            # Directly compare extracted weights with the input weights.
            if file_weight_A == weights[0] and file_weight_B == weights[1]:
                return os.path.join(folder_path, filename)
    return None