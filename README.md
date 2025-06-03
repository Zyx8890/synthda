# AutoSynthDa, the final stage for Project SynthDa  
Pose-Level Synthetic Data Augmentation for Action Recognition
=============================================================

AutoSynthDA is an open-source toolkit that generates class-balanced,
kinematically valid video clips by **interpolating human poses** rather than
rendering full photorealistic frames. The framework is designed to fill
minority-class gaps in action-recognition datasets with minimal compute and no
extra sensors.

By: Megani Rajendran, Chek Tien Tan, Aik Beng Ng, Joey Lim Jun Feng, Triston Chan Sheen, Simon See  
A collaboration between NVAITC-APS and Singapore Institute of Technology    

---

## Why Pose-Level Augmentation?

| Capability                  | Image/Video GANs | **AutoSynthDA (Pose)** |
|-----------------------------|------------------|------------------------|
| Fine-grained motion control | ❌              | ✅                     |
| Independence from textures  | ❌              | ✅                     |
| Render-engine dependency    | High             | None                   |
| Semantic label fidelity     | Medium           | High                   |

*Pose-level synthesis keeps the joint semantics explicit, reduces visual
artifacts, and allows fast generation on commodity GPUs.*

---

## Repository Structure
This section provides a step-by-step guide to setting up the AutoSynthDa pipeline. It covers installing dependencies, placing pretrained models in the correct directories, and testing each component individually. Additionally, common errors and troubleshooting tips are included to help you quickly resolve any setup issues.

📦 Step 1: Install Required Packages
Use `pip` to install the required packages and libraries:

Install all necessary dependencies with:
``pip install -r requirements.txt``

📁 Step 2: Clone Repositories
AutoSynthDa relies on the following repositories.
Make sure you clone and place them correctly in your project structure:

* [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
* [text-to-motion](https://github.com/EricGuo5513/text-to-motion)
* [joints2smpl](https://github.com/wangsen1312/joints2smpl)
* [Blender](https://download.blender.org/release/Blender3.0/)
* [SlowFast](https://github.com/facebookresearch/SlowFast)

⚙️ Step 3: Configuration Setup
Create a .env file in the root directory and add the following paths and keys:

- Your OpenAI GPT API key
- Folder paths to each of the cloned repositories

![image](https://github.com/user-attachments/assets/6f74413c-2e07-49a5-ba37-e9b6213e9ec4)


💾 Step 4: Download Pretrained Models
Each repository requires specific pretrained models or checkpoints. Download all required files from the link below and place them according to the instructions in each repo's section:
👉 [Download Pretrained Files](https://drive.google.com/drive/folders/1E-ZslWCYv07YeGJbQxSFM3GiVt6tKrNk?usp=sharing)

⚠️ Before running the full pipeline, test each repository individually to ensure everything is working.

---

## 🎯 `StridedTransformer-Pose3D`
The StridedTransformer is used to generate Human3.6M (H36M) 3D pose estimations from a real-world video input.

### 📁 Setup & Testing

- Place the **2 `.pth` files** in the following directory:

`./checkpoint/pretrained/`

![StridedTransformer pretrained](https://github.com/user-attachments/assets/43dae193-c4a3-470d-a8ad-69bcdf2694a2)

- Place another **2 `.pth` files** in the following directory:

`./demo/lib/checkpoint/`

![StridedTransformer demo checkpoint](https://github.com/user-attachments/assets/cec84dba-3131-45ae-9a1c-1fb603cbc328)

**How to Test the Repository**
Run the following command:

`python demo/vis.py --video sample_video.mp4`

Video Path:

`StridedTransformer-Pose3D/demo/video/sample_video.mp4`

Output Folder Created:

`StridedTransformer-Pose3D/demo/output/sample_video/`

### 🔧 Troubleshooting

❌ **Error:** `model_path` referenced before assignment
This error might occur due to one or more of the following reasons:

- 📹 **Video Quality Issues**
  - The video resolution is too low
  - The person is not clearly visible or does not appear in the video

- 🎬 **Video Transition**
  - A scene change or transition disrupts the tracking

- 📁 **Model Path Error**
  - The pretrained model is missing or not placed in the correct folder

📷 **Screenshot of Error**
![image](https://github.com/user-attachments/assets/69eab83f-2312-493a-8a28-49f092b14845)

✅ **Suggested Solution**

To resolve this issue, ensure the video shows a **single continuous action** with the **full body of the person visible throughout**. You may need to crop or split videos that contain transitions or multiple angles.

✂️ **Example Fix: **

**❌ Original Video (with angle transition):**  
The original video contains a shift in camera angle, which can disrupt pose tracking.

https://github.com/user-attachments/assets/95ece067-74c9-48a5-a96c-077a5f512be6

**✅ Fixed Videos (split by angle):**  
Splitting the video into segments with consistent camera angles can significantly improve pose tracking accuracy.

1. First angle:

   https://github.com/user-attachments/assets/f4ad8cfd-bec4-4c9f-a8d7-1b98d5b460f6

2. Second angle:

   https://github.com/user-attachments/assets/f243c2b7-34c4-47dd-b3d9-cf9e29b981fc

---


## 🎬 `text-to-motion`
The text-to-motion is used to generate a 3D pose estimations from a text input.

### 📁 Setup & Testing

- Replace the existing `checkpoints` file with a **folder** named `checkpoints`.
- Inside that folder, place the two pre-trained models:

`./checkpoints/kit/` and `./checkpoints/t2m/`

![text-to-motion checkpoints](https://github.com/user-attachments/assets/490c9bbb-c03d-4000-a5b2-b34c8bd1ad99)

**How to Test the Repository**
Run the following command:

`python gen_motion_script.py --name Comp_v6_KLD01 --text_file input.txt --repeat_time 1`

The expected output should create a folder called C000 containing a .npy file in the eval_results folder:

`text-to-motion/eval_results/t2m/Comp_v6_KLD01/default/animations/C000/gen_motion_00_L044_00_a.npy`
📌 Note: The folder name (e.g., C000) and filename may differ.

### 🔧 Troubleshooting

❌ **Error:** Deprecated NumPy version
This error occurs due to the use of deprecated attributes in newer versions of NumPy.

📷 **Screenshot of Error**
![image](https://github.com/user-attachments/assets/5b35fdc4-5282-4785-8bbb-25b5044d6e5e)

✅ **Suggested Solution**

Replace both instances of `np.float` in `./common/quaternion.py` — use `float` on **Line 11**, and `np.float64` on **Line 13**.
![image](https://github.com/user-attachments/assets/8009de8d-56c7-4c07-9a01-5441c25d764d)

❌ **Error:** Error faced with ax.lines
This error occurs because `ax.lines` and `ax.collections` are **read-only** properties and cannot be directly assigned.

📷 **Screenshot of Error**
![image](https://github.com/user-attachments/assets/a17384fa-af8e-41f2-916e-93d1e7c4d56b)

✅ **Suggested Solution**
Adjust the `ax.lines` assignment lines inside the `update(index)` function located in `./utils/plot_script.py`.
![image](https://github.com/user-attachments/assets/bca8b006-3f32-405b-8ae2-773bbc63bf4b)

❌ **Error:** Unable to Recognize `spacy` Package
This error occurs when the required SpaCy language model is not installed. It will state that it is "unable to find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory. "

✅ **Suggested Solution**
python -m spacy download en_core_web_sm

❌  **Error:** Unable to Recognize `ffmpeg` Package
This error occurs when the required ffmpeg package is not installed

✅ **Suggested Solution**
conda install -c conda-forge ffmpeg

---

## ⚙️ `joints2smpl`
The joints2smpl is used to attach the generated joints to the SMPL character body.

### 📁 Setup & Testing

- **SMPL models** must be placed inside the following path:

`./smpl_models/smpl/`

![joints2smpl model placement](https://github.com/user-attachments/assets/d50e2d29-416a-43eb-9cfa-031a98d7ad70)

**How to Test the Repository**

Run the following command:

`python fit_seq.py --files test_motion2.npy`

Input .npy File Location:

`joints2smpl/demo/demo_data/test_motion2.npy`

Output Folder with .obj Files:

`joints2smpl/demo/demo_results/test_motion2/`

### 🔧 Troubleshooting

❌  **Error:** ImportError: cannot import name 'bool' from 'numpy'
This occurs due to deprecated usage of `np.bool`, which has been removed in recent versions of NumPy.

✅ **Suggested Solution**  
Navigate to the location where your `chumpy` package is installed by typing:

`pip show chumpy`

![image](https://github.com/user-attachments/assets/705a4450-6a53-436e-ab88-e2a142f07243)

Go to the `__init__.py` file inside the `chumpy` directory and replace line 11 with lines 13 to 19, as shown in the image below:

![image](https://github.com/user-attachments/assets/76ba31ea-00ee-4774-8ffe-421e7cd1c70b)

---

## 🎬 `Blender`
The Blender library is used to generate a video from the `.ply` objects.

### 📁 Setup & Testing 
Download **Blender version 3.0.0** from the [official Blender website](https://download.blender.org/release/Blender3.0/) or use the command below to download it directly:

`wget https://download.blender.org/release/Blender3.0/blender-3.0.0-linux-x64.tar.xz`

![image](https://github.com/user-attachments/assets/131e11e2-043f-4e91-8767-452c059c5665)

After downloading, unzip the folder using:

`tar -xf blender-3.0.0-linux-x64.tar.xz`

**Blender Setup Files**

- Place your animation_pose.py script into the Blender directory (where ./blender is run).

- Create a angleInput.txt file in the same directory. Inside this file, specify the desired camera angle (e.g., "front"). This determines the camera viewpoint used when rendering the video.

![image](https://github.com/user-attachments/assets/84ed122b-e429-4f68-b3de-26f0d6adae2d)

**How to Test the Blender Tool**

Navigate into the extracted folder and launch Blender using `./blender`. If Blender launches successfully, the installation is complete.

![image](https://github.com/user-attachments/assets/5fd06727-9439-4f60-af8e-7db2db0cf907)

Alternatively, if the `.ply` object folder has already been created, you can convert it into a video directly using the following command:

`./blender -b -P animation_pose.py -- --name <name_of_folder_containing_ply_files>`

---

📌 For detailed model download instructions, refer to the **documentation pages** of each repository.


## 🔧 General Troubleshooting
❌ Error: File path of the different repos are not found
This error occurs due to the codebase's inability to retrieve the variables from the `.env` file.

**✅ Suggested Solution**  
Add the variables directly into the codebase or create a separate Python file to store the folder path variables.

---

❌ Error: Python not recognized
This error occurs when the environment is set up to run python files using `python3 test.py` instead of `python test.py`. The repository expects Python files to be run with the `python` command. If `python` is not mapped to `python3`, the script may fail to execute correctly.

**✅ Suggested Solution**  
Reassign the `python` command to point to `python3` using the following command:

`sudo ln -s $(which python3) /usr/local/bin/python`



