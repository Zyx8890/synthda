# AutoSynthDa, the final stage for Project SynthDa  
Pose-Level Synthetic Data Augmentation for Action Recognition
=============================================================

AutoSynthDA is an open-source toolkit that generates class-balanced,
kinematically valid video clips by **interpolating human poses** rather than
rendering full photorealistic frames. The framework is designed to fill
minority-class gaps in action-recognition datasets with minimal compute and no
extra sensors.

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

