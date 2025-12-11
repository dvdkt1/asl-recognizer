# ASL Alphabet Recognition — Embedded AI Project  
Real-Time, Privacy-First, On-Device Sign Language Classification

## Overview
This project implements a fully client-side American Sign Language (ASL) alphabet recognizer that runs entirely in the browser using MediaPipe Hands for landmark detection and a lightweight TensorFlow.js MLP classifier for real-time inference.

All computation happens on the user’s device — no frames are ever sent to a server, ensuring low latency, full privacy, and instant accessibility from any modern browser.

---

# 1. Problem Definition

## What We Built
A real-time ASL alphabet interpreter that:
- Uses webcam input  
- Extracts 21 MediaPipe 3D hand landmarks  
- Normalizes them into a 63-dimensional feature vector  
- Classifies them using a compact neural network running in-browser  
- Displays predictions with FPS, latency, and metrics logging

## What Problem It Solves
Most ASL translation tools use cloud inference, which introduces:
- High network latency  
- Privacy concerns from sending video off-device  
- Dependency on stable connectivity  

This project avoids all of these issues by keeping inference 100% local.

## Dataset / Sensor Data
- **Sensor:** Standard webcam  
- **Features:** (x, y, z) coordinates for 21 hand landmarks (63 features)  
- **Dataset:**  
  - Self-collected samples through built-in “Collect Mode”  
  - Supplemental ASL alphabet samples from public datasets  

## Why It Matters
- Accessible ASL recognition for beginners and learners  
- Privacy-first assistive technology  
- Real-time inference with no backend dependency  
- Enables edge deployment on any device with a browser  

---

# 2. Model & Method Summary

## Model Architecture
A tiny, efficient MLP optimized for browser inference:

| Layer | Details |
|--------|---------|
| Input | 63 features (21 landmarks × 3 dims) |
| Dense 1 | 64 units, ReLU + Dropout(0.2) |
| Dense 2 | 32 units, ReLU |
| Output | 26-class softmax |

- **Model Size:** ~50 KB  
- **Parameters:** ~5,000  
- **Training Framework:** Python/Keras  
- **Deployment:** TensorFlow.js format  

---

## Preprocessing Pipeline
Identical in Python and TypeScript:

1. **Wrist-Centering** — subtract wrist landmark from all points  
2. **Max-Distance Scaling** — normalize by the maximum Euclidean distance  
3. **Flattening** — convert 21×3 → 63-dimensional vector  

This guarantees invariance to:
- Hand position  
- Distance from camera  
- Minor viewpoint shifts  

---

## Deployment Pipeline
- **Next.js 16** frontend  
- **MediaPipe Hands** for real-time landmark extraction  
- **TensorFlow.js (WebGL backend)** for GPU inference  
- **Canvas overlay** for predictions and UI  
- Entire system runs **fully on device** with no server calls

---

# 3. Results (TODO: Update this later)

## Task Performance
| Metric | Value |
|--------|--------|
| Validation Accuracy | ~98% |
| Epochs | 50 |
| Loss | Sparse Categorical Crossentropy |

---

## System Metrics (Browser Inference)
Tested on Chrome Desktop (WebGL):

| Metric | Result |
|--------|--------|
| p50 Latency | ~20 ms |
| FPS | 25–35 FPS |
| Time-to-First-Inference (TTFI) | < 2 seconds |
| Model Size | ~50 KB |
| Bundle Size | < 10 MB |

The model meets all real-time interaction requirements.

---

# 4. Discussion

## What Worked
- Landmark-based approach reduced model size massively  
- Fully on-device inference eliminated privacy and latency concerns  
- MediaPipe Hands delivered high-quality landmarks at real-time FPS  
- GPU acceleration via WebGL kept latency low  

## Challenges (Update This)
- Some ASL letters are visually similar (M, N, R)  
- Predictions can flicker due to frame-by-frame inference  
- Mobile browsers occasionally fall back to CPU mode  

## Future Work
- Add temporal smoothing (moving average / voting)  
- Quantize model to Float16 or INT8 for mobile performance  
- Extend to dynamic signs using sequence modeling  
- Package as a PWA for offline capability  

---

# 5. How to Run

## Install
```bash
npm install
```
## Start Development Server
```bash
npm run dev
# Runs at: http://localhost:3000
```
## Usage
- Allow webcam access 
- Present ASL hand shapes
- View predictions overlaid in real time
- Use COLLECT mode to gather new labeled samples


# 6. Training the Model 
## Prepare Dataset 
- Place asl_dataset.json next to train_model.py

## Train
```bash
python train_model.py
```

## Convert to TensorFlow.js Format
```bash
tensorflowjs_converter \
  --input_format=keras \
  model_v2.h5 \
  public/model_v2/
```

# 7. Repo Structure (Update this with final repo structure(make sure it is ready to ))
```bash
.
├── src/
│   ├── app/
│   ├── components/
│   │   └── CameraView.tsx          # Inference loop + preprocessing
│   └── results/results.csv         # Performance log template
├── public/
│   └── model_v2/                   # TFJS model files
├── train_model.py                  # Training script
├── model_v2.h5                     # Keras FP32 model
├── confusion_matrix.png            # Evaluation output
├── initial_batch.json              # Sample dataset
└── README.md

```

# 8. Model Artifacts Included (Update this if needed)
- model_v2.h5 — FP32 Keras model
- public/model_v2/model.json + shards — TFJS deployment model
- classes.json — label map
- confusion_matrix.png — visualization
- results.csv — system metrics log template

# 9. Demo Video (TODO put link here)

- Put link here


# 10. Deployment Device Setup (Put what you used a with cuda and how it is feasable to deploy )

# 11. License (Put license here)
- license 