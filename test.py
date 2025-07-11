import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import random
from datetime import datetime
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from facenet_pytorch import MTCNN

# =============================================================================
# ===  مشترک SETUP (COMMON SETUP) ==============================================
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {DEVICE}")

# =============================================================================
# === TASK A: GENDER CLASSIFICATION ===========================================
# =============================================================================

# --- Task A: Label Maps ---
idx2label_A = {0: "male", 1: "female"}
label2idx_A = {v: k for k, v in idx2label_A.items()}

# --- Task A: Transforms ---
transform_A = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Task A: MTCNN Setup ---
mtcnn_A = MTCNN(keep_all=True, device=DEVICE)

# --- Task A: Face Crop Logic ---
def crop_central_face(image, padding_ratio=0.45):
    """Detects all faces and crops the one closest to the image center."""
    boxes, _ = mtcnn_A.detect(image)
    if boxes is None or len(boxes) == 0:
        return image  # Fallback to original image if no face is detected
    img_w, img_h = image.size
    img_center = np.array([img_w / 2, img_h / 2])
    face_centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for (x1, y1, x2, y2) in boxes])
    dists = np.linalg.norm(face_centers - img_center, axis=1)
    closest_idx = np.argmin(dists)
    x1, y1, x2, y2 = boxes[closest_idx]
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding_ratio, h * padding_ratio
    x1 = max(int(x1 - pad_w), 0)
    y1 = max(int(y1 - pad_h), 0)
    x2 = min(int(x2 + pad_w), img_w)
    y2 = min(int(y2 + pad_h), img_h)
    return image.crop((x1, y1, x2, y2))

# --- Task A: Load Model ---
def load_model_A(model_path):
    """Loads a ResNet18 model for gender classification."""
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# --- Task A: Grid Display ---
def show_all_A(grid):
    """Displays a grid of images with their predictions."""
    cols = 6
    rows = (len(grid) + cols - 1) // cols
    plt.figure(figsize=(18, 3.5 * rows))
    for i, (img, label, conf) in enumerate(grid):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{label} ({conf}%)", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Task A: Ensemble Inference ---
def run_task_a_inference(model_paths, model_weights, image_folder, save_csv=True, show_matrix=True):
    """Main function to run the gender classification inference."""
    print("\n" + "="*70)
    print("🚀 STARTING TASK A: GENDER CLASSIFICATION")
    print("="*70)
    
    assert len(model_paths) == len(model_weights), "Mismatch in model paths and weights for Task A"
    
    models_list = [load_model_A(path) for path in model_paths]
    image_folder = Path(image_folder)

    folder_name = image_folder.name.lower()
    if folder_name == "male":
        model_weights = np.array([0.85, 0.15])
        print("🔧 Male folder detected → Using weights: [0.85, 0.15]")
    elif folder_name == "female":
        model_weights = np.array([0.15, 0.85])
        print("🔧 Female folder detected → Using weights: [0.15, 0.85]")
    else:
        model_weights = np.array(model_weights)
        print(f"🔧 Custom folder detected → Using weights: {model_weights}")
    model_weights = model_weights / model_weights.sum()

    image_paths = []
    subdirs = [p for p in image_folder.iterdir() if p.is_dir()]
    if any(subdir.name.lower() in label2idx_A for subdir in subdirs):
        for label in label2idx_A.keys():
            folder = image_folder / label
            if folder.exists():
                image_paths.extend(folder.glob("*"))
    else:
        image_paths = list(image_folder.glob("*"))

    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_paths:
        raise ValueError("No images found for Task A!")

    predictions, true_labels, pred_labels, display_images = [], [], [], []

    for img_path in tqdm(image_paths, desc="Running Task A Inference"):
        img = Image.open(img_path).convert("RGB")
        face = crop_central_face(img)
        tensor = transform_A(face).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.zeros(1, 2).to(DEVICE)
            for model, weight in zip(models_list, model_weights):
                output = model(tensor)
                probs += F.softmax(output, dim=1) * weight
            
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        predictions.append({
            "filename": str(img_path.relative_to(image_folder)),
            "predicted_label": idx2label_A[pred],
            "confidence": round(conf, 4)
        })
        pred_labels.append(pred)

        parent = img_path.parent.name.lower()
        if parent in label2idx_A:
            true_labels.append(label2idx_A[parent])
        elif folder_name in label2idx_A:
            true_labels.append(label2idx_A[folder_name])
        else:
            true_labels.append(None)
        
        resized = face.resize((128, 128))
        display_images.append((resized, idx2label_A[pred], round(conf * 100, 1)))

    if len(true_labels) > 0 and all(t is not None for t in true_labels):
        print("\n📊 Task A Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=["male", "female"]))
        if show_matrix:
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["male", "female"])
            disp.plot(cmap="Blues")
            plt.title("Task A: Confusion Matrix")
            plt.show()
    else:
        print("\n⚠️ No ground truth labels available for Task A performance metrics.")

    if save_csv:
        df = pd.DataFrame(predictions)
        csv_path = "task_a_gender_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved Task A predictions to {csv_path}")

    show_all_A(display_images)
    print("🎉 Task A: Gender Classification Complete!")


# =============================================================================
# === TASK B: FACE RECOGNITION ================================================
# =============================================================================

# --- Task B: Config ---
IMG_SIZE_B = 224
EMBED_DIM_B = 128
BATCH_SIZE_B = 64
VISUALIZE_SAMPLES_B = 5

# --- Task B: Transform ---
transform_B = transforms.Compose([
    transforms.Resize((IMG_SIZE_B, IMG_SIZE_B)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Task B: Model ---
class FaceEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM_B):
        super(FaceEncoder, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_layer = nn.Linear(backbone.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)

# --- Task B: Utils ---
def get_user_inputs_B():
    """Get validation folder and find model path for Task B."""
    while True:
        val_path = input("\n📂 [Task B] Enter the face recognition validation/test folder path: ").strip()
        if os.path.exists(val_path) and os.path.isdir(val_path):
            print(f"✅ Validation folder found: {val_path}")
            break
        else:
            print("❌ Invalid path. Please enter a valid directory path.")
    model_path = find_model_path_B()
    return val_path, model_path

def find_model_path_B():
    """Find the face encoder model path robustly."""
    model_filename = "best_face_encoder.pth"
    search_paths = [
        os.path.join(os.getcwd(), "Task_B", model_filename),
        os.path.join(os.getcwd(), model_filename)
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Model found: {path}")
            return path
    
    print(f"❌ Could not find {model_filename} automatically.")
    while True:
        model_path = input(f"🤖 Please enter the full path to {model_filename}: ").strip()
        if os.path.exists(model_path) and model_path.endswith('.pth'):
            print(f"✅ Model found: {model_path}")
            return model_path
        else:
            print("❌ Invalid model path. Please enter a valid .pth file path.")

def load_val_gallery_and_queries_B(val_root):
    """Load gallery and query images from the validation folder."""
    gallery_images, gallery_labels, gallery_paths = [], [], []
    query_images, query_labels, query_paths = [], [], []
    identity_dirs = sorted([d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))])
    
    for identity in tqdm(identity_dirs, desc="Loading Identities"):
        identity_path = os.path.join(val_root, identity)
        for fname in os.listdir(identity_path):
            full_path = os.path.join(identity_path, fname)
            if os.path.isfile(full_path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(full_path).convert("RGB")
                    gallery_images.append(transform_B(img))
                    gallery_labels.append(identity)
                    gallery_paths.append(full_path)
                except Exception as e:
                    print(f"⚠️ Error loading gallery image {full_path}: {e}")

        distortion_folder = os.path.join(identity_path, 'distortion')
        if os.path.isdir(distortion_folder):
            for fname in os.listdir(distortion_folder):
                full_path = os.path.join(distortion_folder, fname)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = Image.open(full_path).convert("RGB")
                        query_images.append(transform_B(img))
                        query_labels.append(identity)
                        query_paths.append(full_path)
                    except Exception as e:
                        print(f"⚠️ Error loading query image {full_path}: {e}")
    print(f"📊 Loaded {len(gallery_images)} gallery images and {len(query_images)} query images.")
    return gallery_images, gallery_labels, gallery_paths, query_images, query_labels, query_paths

def show_match_B(query_path, matched_path, pred, actual, score):
    """Display query and matched images side by side with correctness indication."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    try:
        ax[0].imshow(Image.open(query_path))
        ax[0].set_title(f"Query: {actual}")
        ax[0].axis('off')
        ax[1].imshow(Image.open(matched_path))
        ax[1].set_title(f"Matched: {pred}\nSimilarity: {score:.3f}")
        ax[1].axis('off')
        
        if pred == actual:
            fig.patch.set_facecolor('lightgreen')
            fig.suptitle("✅ CORRECT MATCH", fontsize=14, fontweight='bold')
        else:
            fig.patch.set_facecolor('lightcoral')
            fig.suptitle("❌ INCORRECT MATCH", fontsize=14, fontweight='bold')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    except Exception as e:
        print(f"Error displaying images: {e}")

def print_detailed_metrics_B(y_true, y_pred):
    """Print a comprehensive set of evaluation metrics for Task B."""
    print("\n" + "="*60)
    print("📊 TASK B: DETAILED EVALUATION METRICS")
    print("="*60)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    return accuracy, precision, recall, f1

def save_results_to_csv_B(y_true, y_pred, sim_scores, q_paths, g_paths, metrics):
    """Save detailed recognition results to a timestamped CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"task_b_face_recognition_results_{timestamp}.csv"
    print(f"\n💾 Saving results to {csv_filename}")
    
    results_data = []
    for i in range(len(y_true)):
        is_correct = y_true[i] == y_pred[i]
        results_data.append({
            'Query_Path': q_paths[i],
            'True_Identity': y_true[i],
            'Predicted_Identity': y_pred[i],
            'Matched_Gallery_Path': g_paths[i],
            'Confidence_Score': round(sim_scores[i], 4),
            'Match_Status': "CORRECT" if is_correct else "INCORRECT"
        })
    df = pd.DataFrame(results_data)
    
    # Add summary
    summary_data = {
        'Query_Path': '--- SUMMARY ---',
        'True_Identity': f'Accuracy: {metrics[0]:.4f}',
        'Predicted_Identity': f'Precision: {metrics[1]:.4f}',
        'Matched_Gallery_Path': f'Recall: {metrics[2]:.4f}',
        'Confidence_Score': f'F1-Score: {metrics[3]:.4f}',
        'Match_Status': ''
    }
    df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)
    df.to_csv(csv_filename, index=False)
    print(f"✅ Results saved successfully.")
    return csv_filename

# --- Task B: Main Inference ---
def run_task_b_inference():
    """Main function to run the face recognition inference."""
    print("\n" + "="*70)
    print("🚀 STARTING TASK B: FACE RECOGNITION")
    print("="*70)
    
    val_path, model_path = get_user_inputs_B()
    
    try:
        model = FaceEncoder().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    g_imgs, g_labels, g_paths, q_imgs, q_labels, q_paths = load_val_gallery_and_queries_B(val_path)
    if not g_imgs or not q_imgs:
        print("❌ No gallery or query images found. Please check the folder structure."); return

    print(f"\n🧠 Encoding gallery of {len(g_imgs)} images...")
    g_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(g_imgs), BATCH_SIZE_B), desc="Encoding Gallery"):
            batch = torch.stack(g_imgs[i:i+BATCH_SIZE_B]).to(DEVICE)
            g_embeds.append(model(batch).cpu())
    g_embeds = torch.cat(g_embeds, dim=0).numpy()

    print(f"\n🎯 Matching {len(q_imgs)} query images...")
    y_true, y_pred, sim_scores, g_sample_paths = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(q_imgs)), desc="Matching Queries"):
            q_img = q_imgs[i].unsqueeze(0).to(DEVICE)
            q_emb = model(q_img).squeeze(0).cpu().numpy()
            sims = g_embeds @ q_emb
            best_idx = np.argmax(sims)
            
            y_true.append(q_labels[i])
            y_pred.append(g_labels[best_idx])
            sim_scores.append(sims[best_idx])
            g_sample_paths.append(g_paths[best_idx])

    metrics = print_detailed_metrics_B(y_true, y_pred)
    csv_filename = save_results_to_csv_B(y_true, y_pred, sim_scores, q_paths, g_sample_paths, metrics)

    print("\n📊 Generating confusion matrix...")
    all_classes = sorted(list(set(y_true + y_pred)))
    if len(all_classes) <= 20:
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, xticklabels=all_classes, yticklabels=all_classes, 
                    cmap='Blues', fmt='d', annot=True if len(all_classes) <= 10 else False)
        plt.xlabel("Predicted Identity"); plt.ylabel("Actual Identity")
        plt.title("Task B: Confusion Matrix"); plt.tight_layout(); plt.show()
    else:
        print(f"⚠️ Too many classes ({len(all_classes)}) to display confusion matrix clearly.")

    visualize = input(f"\n🖼 Do you want to visualize {VISUALIZE_SAMPLES_B} sample matches? (y/n): ").strip().lower()
    if visualize in ['y', 'yes']:
        indices = list(range(len(y_true))); random.shuffle(indices)
        for i in range(min(len(indices), VISUALIZE_SAMPLES_B)):
            idx = indices[i]
            show_match_B(q_paths[idx], g_sample_paths[idx], y_pred[idx], y_true[idx], sim_scores[idx])

    print(f"\n🎉 Task B: Face Recognition Complete!")
    print(f"📁 Detailed results saved in: {csv_filename}")


# =============================================================================
# === SCRIPT ENTRY POINT ======================================================
# =============================================================================
if __name__ == "__main__":
    try:
        # --- Run Task A ---
        task_a_image_folder = input("📂 [Task A] Enter path to gender validation or test folder: ").strip()
        # Define model paths relative to the script's location
        BASE_DIR = Path(__file__).resolve().parent
        MODEL_1_PATH = BASE_DIR / "Task_A" / "resnet18_M1_state_dictionary.pth"
        MODEL_2_PATH = BASE_DIR / "Task_A" / "resnet18_M2_state_dictionary.pth"

        # Check if model files exist
        if not MODEL_1_PATH.exists() or not MODEL_2_PATH.exists():
             print("❌ Error: Task A model files not found!")
             print(f"Ensure '{MODEL_1_PATH.name}' and '{MODEL_2_PATH.name}' are in a 'Task_A' subfolder.")
        else:
             task_a_model_paths = [MODEL_1_PATH, MODEL_2_PATH]
             task_a_model_weights = [0.6, 0.4] # Default weights
             run_task_a_inference(task_a_model_paths, task_a_model_weights, task_a_image_folder)
        
        # --- Run Task B ---
        run_task_b_inference()

    except KeyboardInterrupt:
        print("\n\n⏹️ Script interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()