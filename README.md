#  Cloud Segmentation – Little Place Labs Assignment

This repository contains my implementation of the **Cloud Segmentation Assignment** provided by **Little Place Labs**.  
The objective is to experiment with multiple methods — from classical image processing to deep learning — for segmenting clouds in satellite imagery.

---

## Experiments

All experiments are numbered and implemented as separate scripts:

| Script | Method | Description |
|--------|--------|-------------|
| **5.1.manual_thresholding.py** | Manual Thresholding | Basic segmentation using fixed intensity thresholds. |
| **5.2.brightness_masking.py** | Brightness Masking | Identifies cloud pixels based on brightness (whiteness assumption). |
| **5.3.flood_filling.py** | Flood Filling | Region growing starting from brightest pixel, expanding until threshold. |
| **5.4.processed_flood_filling.py** | Processed Flood Filling | Enhanced flood filling with preprocessing for smoother masks. |
| **5.5.fmask\*.py** | Fmask | Implementation of the well-known **Fmask algorithm** for cloud masking. |
| **5.6.cloudnet.py** | Cloud-Net | Deep learning model (U-Net variant) trained for cloud segmentation. |
| **5.7.aunet.py** | Attention U-Net | Advanced U-Net with attention mechanism for better feature focus. |

---

