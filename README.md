# AutoSynthDa, under Project SynthDa  
Pose-Level Synthetic Data Augmentation for Action Recognition (Research Purposes Only)  
=============================================================

#### Licensed under NSCLv2, where users can use this noncommercially (which means research or evaluation purposes only) on NVIDIA Processors.  

![image](https://github.com/user-attachments/assets/1fde62ce-67a6-4673-9341-78da4daa31e4)
SynthDa (aka AutoSynthDa) is an open-source toolkit that generates class-balanced,
kinematically valid video clips by automatically **interpolating human poses** rather than
rendering full photorealistic frames. The framework is designed to mitigate the issue of imbalanced datasets by creating synthetic data for minority action classes in action-recognition datasets without the need for additional sensors/modality. This only uses RGB videos to generate synthetic videos.  

Each component of our proposed framework, can be swapped with models of your choice / components of your choice.  For the augmentation optimization loop, we have used [action recognition net](https://github.com/NVIDIA/tao_tutorials/tree/main/notebooks/tao_launcher_starter_kit/action_recognition_net) from NVIDIA TAO Toolkit.  We provided each of the components, to be used individually or stringed together/automated yourself for your specific use-case. Our purpose is to enable improved synthetic data generation for human actions.

### See our [Wiki Page](https://github.com/NVIDIA/synthda/wiki) for more customization options, and the full [Set Up instructions](https://github.com/NVIDIA/synthda/wiki/Setting-Up-SynthDa).  

Project By:   
Megani Rajendran, Chek Tien Tan, Aik Beng Ng, Indriyati Atmosukarto, Joey Lim Jun Feng, Triston Chan Sheen, Simon See  
A collaboration between NVAITC-APS and Singapore Institute of Technology   
   

---



## Why Pose-Level Augmentation?

| Capability                  | Image/Video GANs | **AutoSynthDa (Pose)** |
|-----------------------------|------------------|------------------------|
| Fine-grained motion control | Low-Medium       | ✅ Improved           |
| Independence from textures  | Low              | ✅ Improved           |
| Maintaining Semantic labels | Medium           | ✅ Improved           |

*Pose-level synthesis potentially keeps the joint semantics explicit, reduces visual
artifacts, and allows customizable generation speed and quality on NVIDIA GPUs.*  

Note that this code has only been developed and tested with NVIDIA Processors  

---

## Repository Structure
## Minimal Setup Checklist

| Step | Command / Action | Notes |
|------|------------------|-------|
| **1&nbsp;Install deps** | `pip install -r requirements.txt` | CUDA-enabled PyTorch recommended |
| **2&nbsp;Clone sub-repos** | `git clone` the five required repos <br>*(see list below)* | Keep the folder names unchanged |
| **3&nbsp;Create `.env`** | Add your OpenAI key and paths to each repo | No key provided, you will need to add your own |
| **4&nbsp;Download models** | Grab all checkpoints from [**setup wiki**](https://github.com/NVIDIA/synthda/wiki/Setting-Up-SynthDa) | Place files exactly as instructed |
| **5&nbsp;Smoke-test** | Run each repo’s test script once | Fail-fast before running the full pipeline |

### Required External Repositories

```text
StridedTransformer-Pose3D/
text-to-motion/
joints2smpl/
SlowFast/
Blender-3.0.0/          (binary drop)
