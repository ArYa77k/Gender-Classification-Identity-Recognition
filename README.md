
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

## 🔧 Setup Instructions & Environment Dependencies

1. Create a virtual environment (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # or .\env\Scripts\activate on Windows
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure your environment supports GPU acceleration (CUDA) if available for faster inference.

---

## 📁 Folder Structure & Input Format

```
CHEHRA/
│
├── Task_A/                                 # Gender Classification Task
│   ├── resnet18_M1_state_dictionary.pth    # Model 1 weights
│   ├── resnet18_M2_state_dictionary.pth    # Model 2 weights
│
├── Task_B/                                 # Face Recognition Task
│   ├── best_face_encoder.pth               # Best trained encoder
│
├── test.py                                 # Entry-point for evaluation
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

### Expected Input Format for Inference

- **Task A: Gender Classification**
  ```
  your_folder/
  ├── male/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── female/
      ├── img1.jpg
      ├── img2.jpg
  ```

- **Task B: Face Recognition**
  ```
  val_set/
  ├── identity_1/
  │   ├── gallery_img1.jpg
  │   └── distortion/
  │       └── query_img1.jpg
  ├── identity_2/
      ├── gallery_img2.jpg
      └── distortion/
          └── query_img2.jpg
  ```

---

## ▶️ Example Command to Run `test.py`

```bash
python test.py
# You will be prompted to enter the path to the validation/test folder.
```

---

## 🧠 Model Architectures and Evaluation

### Task A – Gender Classification

- **Model 1**: ResNet18 with full-face inputs and augmentation
- **Model 2**: MT-CNN cropped face with ResNet18 ensemble

📊 **Validation Results (Central Face Crop Ensemble):**
- Accuracy: **99.21%**
- Precision: **99.18%**
- Recall: **99.19%**
- F1 Score: **99.18%**

### Task B – Face Recognition

- **Model**: ResNet18-based Face Encoder trained using Triplet Loss

📊 **Validation Results:**
- Top-1 Accuracy: **99.26%**
- Precision: **99.27%**
- Recall: **99.27%**
- F1 Score: **99.27%**

---

## 🔁 Reproducing Submitted Results

1. Make sure you place your validation/test folder as per the format above.
2. Run `python test.py` from the root directory.
3. Enter your folder path when prompted.
4. Inference results and metrics will be displayed and saved as CSV.

**Note:** 
- The ensemble model adjusts weights dynamically based on input folder class (e.g., "male" → [0.85, 0.15])
- All performance metrics (Precision, Recall, Accuracy, F1) are auto-calculated.

---

## ✉️ Contact

- Name: Md Haaris Hussain  
- Email: mdhaarishussain@gmail.com

---

**Good luck to everyone! May the best model win 🏆**
