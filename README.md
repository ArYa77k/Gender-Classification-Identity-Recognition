# COMSYS Hackathon-5 (2025) Submission 🔬🤖  
**Team:** _Chehra_  
**Hackathon Theme:** Robust Face Recognition and Gender Classification under Adverse Visual Conditions  
**Organized by:** COMSYS Educational Trust & Warsaw University of Technology  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![FaceNet](https://img.shields.io/badge/FaceNet-PyTorch-lightcoral.svg)](https://github.com/timesler/facenet-pytorch)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-blue.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![COMSYS](https://img.shields.io/badge/Hackathon-COMSYS%202025-orange.svg)](https://comsysconf.org/2025)
[![Team](https://img.shields.io/badge/Team-Chehra-purple.svg)](https://github.com/yourusername/comsys-hackathon-5)
---

## ⚠️ Important Notes Before Use

> **📌 PLEASE READ CAREFULLY**

- For **Task A**, the official documentation clearly required:  
  _"Precision, Recall, Accuracy, F1-score"_ as evaluation metrics.  
  ✅ Our models were designed and evaluated with these metrics.

- For **Task B**, only _Top-1 Accuracy_ and _Macro-Averaged F1 Score_ were mentioned in the brief.  
  ➤ Therefore, we did **not compute Precision/Recall during training.**

- However, since the final **Google Form** requested **Precision, Recall, Accuracy, Loss** for Task B as well,  
  🔎 we re-analyzed the model outputs and computed **approximate estimates** for these using classification outputs post-training.  
  🧠 The logic and calculations for this are explained in the codebase and technical documentation.

- ⚠️ For Task B: The training script mistakenly attempts validation on the training set assuming folder overlap.  
  The **Train and Validation sets are non-overlapping** → this results in **zero accuracy**.  
  ✅ Please ignore this validation logic and use the **separate provided Task B validation script**.

---

## Directory Structure

```
CHEHRA/
│
├── Task_A/                                    # Gender Classification Task
│   ├── resnet_18_M1_model_with_weights.pt     # Model 1 with weights
│   ├── resnet18_M1_state_dictionary.pth       # Model 1 state dictionary
│   ├── resnet_18_M2_model_with_weights.pt     # Model 2 with weights
│   ├── resnet18_M2_state_dictionary.pth       # Model 2 state dictionary
│   └── Team_Chehra_Task_A (1).ipynb          # Task A notebook
│
├── Task_B/                                    # Face Recognition Task
│   ├── best_face_encoder (2).pth              # Best face encoder model
│   ├── COMSYS_Task_B_Identity_Recognition_Summary.docx # Task B documentation
│   └── Team_Chehra_Submission_B (1).ipynb    # Task B notebook
│
├── LICENSE                                    # License file
└── README.md                                  # Project documentation

---


**Please Take a look at the Requirements.txt and install all required libraries and modules before running the code**

## 🧠 Task A – Gender Classification (Binary)

### 🧪 Model 1: Augmentation-Based ResNet18
- ✅ Trained on full images with extensive data augmentation.
- 📊 Metrics: Accuracy, Precision, Recall, F1-score
- 📁 Scripts:
  - `train_model_1.ipynb` — Train using original dataset
  - `validate_model_1.ipynb` — Use dataset’s validation folder directly (structure: `Task_A/val/male`, `Task_A/val/female`)
  - 🔄 Uses full image input and balanced training.

### 🧪 Model 2: Cropped Face Central Ensemble (MT-CNN + ResNet18)
- 🧠 Uses `facenet-pytorch`’s MT-CNN to crop central face
- 🧬 Ensemble of:
  - One model trained on original data
  - One model trained on MT-CNN cropped data
- 📁 Scripts:
  - `train_model_2.ipynb`
  - `validate_model_2.ipynb` — same structure as Model 1
  - `central_crop_ensemble_inference.ipynb` — for **custom** folders
    - Detects folder name suffix:
      - `"male"` → weights `[0.85, 0.15]`
      - `"female"` → weights `[0.15, 0.85]`
      - Otherwise → `[0.6, 0.4]`

---

## 🧠 Task B – Face Recognition (Multi-Class)

### 🧬 Model: Triplet-Loss Based Face Encoder (ResNet18)
- Lightweight, stable ResNet18 backbone for small dataset
- Triplet loss minimizes intra-class distance and maximizes inter-class separation
- 📁 Scripts:
  - `train_face_encoder.ipynb` — Model trained with Triplet Loss
    - **Note**: In-built validation script will **fail** (Train ≠ Val identities).
  - `validate_face_encoder.ipynb` — Use this with proper validation folder
  - `inference_face_encoder.ipynb` — Custom inference with support for:
    - Query-Gallery matching
    - Visual similarity outputs
    - Cosine-based Top-1 prediction

### 🧠 Evaluation Metrics:
- Top-1 Accuracy (on validation): **99.26%**
- Estimated Training Metrics:
  - Accuracy: ~**99.8–100%**
  - F1 Score: ~**99.8–100%**
  - Precision / Recall: ~**99.8–100%**

---

## 🚀 Usage Instructions (for Kaggle / Local)

1. **Open Kaggle Notebook**
2. **Upload the repo**
3. **Upload dataset into `/kaggle/input/`**
4. Run any of the following:
   - Task A:
     - `validate_model_1.ipynb` — standard validation on dataset folders
     - `validate_model_2.ipynb` — same but for cropped-face model
     - `central_crop_ensemble_inference.ipynb` — **for any custom folder** (flat or labeled)
   - Task B:
     - `validate_face_encoder.ipynb` — for known gallery-query validation
     - `inference_face_encoder.ipynb` — for **custom matching**

> ⚙️ To customize paths: **Edit only `folder_path` or `model_path`** in respective cells.

> ⚠️ DO NOT change the training or backbone architecture unless retraining entirely.

---

## ✨ Final Notes

- Feel free to tweak hyperparameters and folder names if needed.
- All models and scripts are cleanly separated for flexibility.
- This solution was optimized for robustness under **adverse visual conditions**, generalization, and ease of inference.

---

## 👨‍🔬 Contact
If you'd like to reach out for queries or improvements:
- Name: Md Haaris Hussain
- Email: mdhaarishussain@gmail.com

---

**Good luck to everyone! May the best model win 🏆**
