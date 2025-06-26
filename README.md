# real-vs-ai-face-detector
CalHacks 2025
# Deepfake Detector

A real-time “Real vs AI-Generated Face” classifier built with PyTorch, FastAPI & MTCNN.  It learns to distinguish genuine portrait photos from those synthesized by modern generative models (e.g. Stable Diffusion, StyleGAN, etc.), and exposes both an HTML + JSON API for drag-&-drop / programmatic inference.

---

## 🚀 Features

- **Two-stage fine-tuning** on a balanced 120 K real / 120 K AI-generated face dataset  
- **Multi-scale attention** Squeeze-and-Excitation head grafted onto VGG-16 backbone  
- **OneCycleLR schedules** with discriminative learning rates & label smoothing  
- **On-the-fly face cropping** via MTCNN for robust, off-center image handling  
- **FastAPI server** with both HTML and `/api/predict` JSON endpoints  
- **Modern drag-&-drop UI** with confidence bar, mobile-friendly styling  

---
