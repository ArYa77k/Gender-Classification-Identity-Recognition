import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torchvision import models, transforms
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import csv
import pandas as pd
from datetime import datetime


from facenet_pytorch import MTCNN

# ---------------- Label Maps ---------------- #
idx2label = {0: "male", 1: "female"}
label2idx = {v: k for k, v in idx2label.items()}

# ---------------- Transforms ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- MTCNN Setup ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# ---------------- Face Crop Logic ---------------- #
def crop_central_face(image, padding_ratio=0.45):
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return image  # fallback
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

# ---------------- Load Model ---------------- #
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# ---------------- Ensemble Inference ---------------- #
def run_ensemble_inference(model_paths, model_weights, image_folder, save_csv=True, show_matrix=True):
    assert len(model_paths) == len(model_weights), "Mismatch in model paths and weights"
    models_list = [load_model(path) for path in model_paths]
    image_folder = Path(image_folder)

    # Auto weight override based on folder name
    folder_name = image_folder.name.lower()
    if folder_name == "male":
        model_weights = np.array([0.85, 0.15])
        print("🔧 Male folder detected → Using weights: [0.85, 0.15]")
    elif folder_name == "female":
        model_weights = np.array([0.15, 0.85])
        print("🔧 Female folder detected → Using weights: [0.15, 0.85]")
    else:
        model_weights = np.array([0.6, 0.4])
        print("🔧 Custom folder detected → Using weights: [0.6, 0.4]")
    model_weights = model_weights / model_weights.sum()

    # Gather image paths
    image_paths = []
    subdirs = [p for p in image_folder.iterdir() if p.is_dir()]
    if any(subdir.name.lower() in label2idx for subdir in subdirs):
        for label in label2idx.keys():
            folder = image_folder / label
            if folder.exists():
                image_paths.extend(folder.glob("*"))
    else:
        image_paths = list(image_folder.glob("*"))

    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_paths:
        raise ValueError("No images found!")

    predictions = []
    true_labels = []
    pred_labels = []
    display_images = []

    for img_path in tqdm(image_paths, desc="Running Inference"):
        img = Image.open(img_path).convert("RGB")
        face = crop_central_face(img)
        tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.zeros(1, 2).to(device)
            for model, weight in zip(models_list, model_weights):
                output = model(tensor)
                probs += F.softmax(output, dim=1) * weight

            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        predictions.append({
            "filename": str(img_path.relative_to(image_folder)),
            "predicted_label": idx2label[pred],
            "confidence": round(conf, 4)
        })
        pred_labels.append(pred)

        parent = img_path.parent.name.lower()
        if parent in label2idx:
            true_labels.append(label2idx[parent])
        elif folder_name in label2idx:
            true_labels.append(label2idx[folder_name])
        else:
            true_labels.append(None)

        resized = face.resize((128, 128))
        display_images.append((resized, idx2label[pred], round(conf * 100, 1)))

    if len(true_labels) > 0 and all(t is not None for t in true_labels):
        print("\n📊 Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=["male", "female"]))
        if show_matrix:
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["male", "female"])
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()
    else:
        print("\n⚠ No ground truth labels available for performance metrics.")

    if save_csv:
        df = pd.DataFrame(predictions)
        df.to_csv("central_face_ensemble_predictions.csv", index=False)
        print("✅ Saved predictions to central_face_ensemble_predictions.csv")

    show_all(display_images)

# ---------------- Grid Display ---------------- #
def show_all(grid):
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


# --- CONFIG ---
IMG_SIZE = 224
EMBED_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
VISUALIZE_SAMPLES = 5  # Number of matches to visualize

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- MODEL ---
class FaceEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super(FaceEncoder, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_layer = nn.Linear(backbone.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        return nn.functional.normalize(x, p=2, dim=1)

# --- UTILS ---
def get_user_inputs():
    """Get validation folder path from user and find model path automatically"""
    print("=" * 60)
    print("🔍 FACE RECOGNITION VALIDATION SCRIPT")
    print("=" * 60)
    
    # Get validation folder path
    while True:
        val_path = input("\n📂 Enter the validation/test folder path: ").strip()
        if os.path.exists(val_path) and os.path.isdir(val_path):
            print(f"✅ Validation folder found: {val_path}")
            break
        else:
            print("❌ Invalid path. Please enter a valid directory path.")
    
    # Find model path automatically
    model_path = find_model_path()
    
    return val_path, model_path

def find_model_path():
    """Find model path robustly across different devices"""
    model_filename = "best_face_encoder.pth"
    
    # Possible locations to search for the model
    search_paths = [
        # Current directory and subdirectories
        os.path.join(os.getcwd(), "Chehra", "Task_B", model_filename),
        os.path.join(os.getcwd(), "chehra", "Task_B", model_filename),
        os.path.join(os.getcwd(), "Chehra", model_filename),
        os.path.join(os.getcwd(), "chehra", model_filename),
        os.path.join(os.getcwd(), model_filename),
        
        # Parent directories
        os.path.join(os.path.dirname(os.getcwd()), "Chehra", "Task_B", model_filename),
        os.path.join(os.path.dirname(os.getcwd()), "chehra", "Task_B", model_filename),
        
        # Common project structure locations
        os.path.join(".", "Chehra", "Task_B", model_filename),
        os.path.join("..", "Chehra", "Task_B", model_filename),
        os.path.join("models", model_filename),
        os.path.join("checkpoints", model_filename),
    ]
    
    # Try each path
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Model found: {path}")
            return path
    
    # If not found in common locations, search recursively
    print("🔍 Searching for model file recursively...")
    for root, dirs, files in os.walk(os.getcwd()):
        if model_filename in files:
            found_path = os.path.join(root, model_filename)
            print(f"✅ Model found: {found_path}")
            return found_path
    
    # If still not found, search parent directories
    parent_dir = os.path.dirname(os.getcwd())
    if parent_dir != os.getcwd():  # Avoid infinite loop
        for root, dirs, files in os.walk(parent_dir):
            if model_filename in files:
                found_path = os.path.join(root, model_filename)
                print(f"✅ Model found: {found_path}")
                return found_path
    
    # If still not found, ask user
    print(f"❌ Could not find {model_filename} automatically.")
    while True:
        model_path = input(f"🤖 Please enter the full path to {model_filename}: ").strip()
        if os.path.exists(model_path) and model_path.endswith('.pth'):
            print(f"✅ Model found: {model_path}")
            return model_path
        else:
            print("❌ Invalid model path. Please enter a valid .pth file path.")

def load_val_gallery_and_queries(val_root):
    """Load gallery and query images from validation folder"""
    gallery_images, gallery_labels, gallery_paths = [], [], []
    query_images, query_labels, query_paths = [], [], []

    identity_dirs = sorted(os.listdir(val_root))
    print(f"📋 Found {len(identity_dirs)} identity directories")
    
    for identity in identity_dirs:
        identity_path = os.path.join(val_root, identity)
        if not os.path.isdir(identity_path):
            continue

        # Load gallery images (main folder)
        for fname in os.listdir(identity_path):
            full_path = os.path.join(identity_path, fname)
            if os.path.isfile(full_path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(full_path).convert("RGB")
                    gallery_images.append(transform(img))
                    gallery_labels.append(identity)
                    gallery_paths.append(full_path)
                except Exception as e:
                    print(f"⚠️ Error loading {full_path}: {e}")

        # Load query images (distortion folder)
        distortion_folder = os.path.join(identity_path, 'distortion')
        if os.path.isdir(distortion_folder):
            for fname in os.listdir(distortion_folder):
                full_path = os.path.join(distortion_folder, fname)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = Image.open(full_path).convert("RGB")
                        query_images.append(transform(img))
                        query_labels.append(identity)
                        query_paths.append(full_path)
                    except Exception as e:
                        print(f"⚠️ Error loading {full_path}: {e}")

    print(f"📊 Loaded {len(gallery_images)} gallery images and {len(query_images)} query images")
    return gallery_images, gallery_labels, gallery_paths, query_images, query_labels, query_paths

def show_match(query_path, matched_path, pred, actual, score):
    """Display query and matched images side by side"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    try:
        ax[0].imshow(Image.open(query_path))
        ax[0].set_title(f"Query: {actual}")
        ax[0].axis('off')
        
        ax[1].imshow(Image.open(matched_path))
        ax[1].set_title(f"Matched: {pred}\nSimilarity: {score:.3f}")
        ax[1].axis('off')
        
        # Add border color based on correctness
        if pred == actual:
            fig.patch.set_facecolor('lightgreen')
            fig.suptitle("✅ CORRECT MATCH", fontsize=14, fontweight='bold')
        else:
            fig.patch.set_facecolor('lightcoral')
            fig.suptitle("❌ INCORRECT MATCH", fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying images: {e}")

def print_detailed_metrics(y_true, y_pred):
    """Print comprehensive evaluation metrics"""
    print("\n" + "="*60)
    print("📊 DETAILED EVALUATION METRICS")
    print("="*60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Print metrics
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Additional statistics
    total_samples = len(y_true)
    correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect_predictions = total_samples - correct_predictions
    
    print(f"\n📈 Sample Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Incorrect predictions: {incorrect_predictions}")
    print(f"   Unique identities: {len(set(y_true))}")
    
    return accuracy, precision, recall, f1

def save_results_to_csv(y_true, y_pred, sim_scores, q_sample_paths, g_sample_paths, accuracy, precision, recall, f1):
    """Save detailed results to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"face_recognition_results_{timestamp}.csv"
    
    print(f"\n💾 Saving results to {csv_filename}")
    
    # Prepare data for CSV
    results_data = []
    
    for i in range(len(y_true)):
        # Extract image names from paths
        query_img_name = os.path.basename(q_sample_paths[i])
        gallery_img_name = os.path.basename(g_sample_paths[i])
        
        # Extract folder names (identity names)
        query_folder = os.path.basename(os.path.dirname(q_sample_paths[i]))
        gallery_folder = os.path.basename(os.path.dirname(g_sample_paths[i]))
        
        # Check if distortion folder
        if query_folder == 'distortion':
            query_folder = os.path.basename(os.path.dirname(os.path.dirname(q_sample_paths[i])))
        
        # Determine if match is correct
        is_correct = y_true[i] == y_pred[i]
        match_status = "CORRECT" if is_correct else "INCORRECT"
        
        # Calculate confidence percentage
        confidence_pct = (sim_scores[i] * 100) if sim_scores[i] > 0 else 0
        
        row_data = {
            'Query_Image': query_img_name,
            'Query_Path': q_sample_paths[i],
            'True_Identity': y_true[i],
            'Predicted_Identity': y_pred[i],
            'Matched_Gallery_Image': gallery_img_name,
            'Matched_Gallery_Path': g_sample_paths[i],
            'Confidence_Score': round(sim_scores[i], 4),
            'Confidence_Percentage': round(confidence_pct, 2),
            'Match_Status': match_status,
            'Is_Correct': is_correct
        }
        
        results_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    
    # Add summary statistics at the end
    summary_data = {
        'Query_Image': '--- SUMMARY STATISTICS ---',
        'Query_Path': '',
        'True_Identity': f'Total Samples: {len(y_true)}',
        'Predicted_Identity': f'Correct: {sum(y_true[i] == y_pred[i] for i in range(len(y_true)))}',
        'Matched_Gallery_Image': f'Incorrect: {sum(y_true[i] != y_pred[i] for i in range(len(y_true)))}',
        'Matched_Gallery_Path': f'Unique Identities: {len(set(y_true))}',
        'Confidence_Score': f'Accuracy: {accuracy:.4f}',
        'Confidence_Percentage': f'Precision: {precision:.4f}',
        'Match_Status': f'Recall: {recall:.4f}',
        'Is_Correct': f'F1-Score: {f1:.4f}'
    }
    
    # Add empty row and summary
    df = pd.concat([df, pd.DataFrame([{}]), pd.DataFrame([summary_data])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Results saved to {csv_filename}")
    print(f"📊 CSV contains {len(results_data)} prediction records")
    
    # Show sample of data
    print(f"\n📋 Sample of saved data:")
    print(df.head(3)[['Query_Image', 'True_Identity', 'Predicted_Identity', 'Confidence_Percentage', 'Match_Status']].to_string(index=False))
    
    return csv_filename
    """Print comprehensive evaluation metrics"""
    print("\n" + "="*60)
    print("📊 DETAILED EVALUATION METRICS")
    print("="*60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Print metrics
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Additional statistics
    total_samples = len(y_true)
    correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect_predictions = total_samples - correct_predictions
    
    print(f"\n📈 Sample Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Incorrect predictions: {incorrect_predictions}")
    print(f"   Unique identities: {len(set(y_true))}")
    
    return accuracy, precision, recall, f1

# --- INFERENCE ---
def run_val_inference():
    """Main inference function"""
    # Get user inputs
    val_path, model_path = get_user_inputs()
    
    print(f"\n📦 Loading model from {model_path}")
    try:
        model = FaceEncoder().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"\n📂 Loading gallery and queries from {val_path}")
    try:
        g_imgs, g_labels, g_paths, q_imgs, q_labels, q_paths = load_val_gallery_and_queries(val_path)
        if len(g_imgs) == 0 or len(q_imgs) == 0:
            print("❌ No images found. Please check the folder structure.")
            return
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return

    # Encode gallery in batches
    print(f"\n🧠 Encoding gallery of {len(g_imgs)} images...")
    g_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(g_imgs), BATCH_SIZE), desc="Encoding gallery"):
            batch = torch.stack(g_imgs[i:i+BATCH_SIZE]).to(DEVICE)
            g_embeds.append(model(batch).cpu())
    g_embeds = torch.cat(g_embeds, dim=0).numpy()

    # Encode query in batches
    print(f"\n🎯 Matching {len(q_imgs)} query images...")
    y_true, y_pred = [], []
    sim_scores, q_sample_paths, g_sample_paths = [], [], []

    with torch.no_grad():
        for i in tqdm(range(len(q_imgs)), desc="Matching queries"):
            q_img = q_imgs[i].unsqueeze(0).to(DEVICE)
            q_emb = model(q_img).squeeze(0).cpu().numpy()

            sims = g_embeds @ q_emb
            best_idx = np.argmax(sims)
            predicted = g_labels[best_idx]
            actual = q_labels[i]

            y_true.append(actual)
            y_pred.append(predicted)
            sim_scores.append(sims[best_idx])
            q_sample_paths.append(q_paths[i])
            g_sample_paths.append(g_paths[best_idx])

    # --- METRICS ---
    accuracy, precision, recall, f1 = print_detailed_metrics(y_true, y_pred)
    
    # --- SAVE TO CSV ---
    csv_filename = save_results_to_csv(y_true, y_pred, sim_scores, q_sample_paths, g_sample_paths, 
                                     accuracy, precision, recall, f1)

    # --- CONFUSION MATRIX ---
    print("\n📊 Generating confusion matrix...")
    all_classes = sorted(list(set(y_true + y_pred)))
    if len(all_classes) <= 20:  # Only show confusion matrix for manageable number of classes
        label_to_idx = {name: i for i, name in enumerate(all_classes)}
        y_true_idx = [label_to_idx[y] for y in y_true]
        y_pred_idx = [label_to_idx[y] for y in y_pred]

        cm = confusion_matrix(y_true_idx, y_pred_idx)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, xticklabels=all_classes, yticklabels=all_classes, 
                   cmap='Blues', fmt='d', annot=True if len(all_classes) <= 10 else False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Too many classes ({len(all_classes)}) to display confusion matrix clearly.")
        print(f"Please refer to the face_recogniton_result.csv ")

    # --- VISUALIZE SAMPLES ---
    visualize = input(f"\n🖼 Do you want to visualize {VISUALIZE_SAMPLES} sample matches? (y/n): ").strip().lower()
    if visualize in ['y', 'yes']:
        print(f"\n🖼 Showing {VISUALIZE_SAMPLES} match samples...")
        indices = list(range(len(y_true)))
        random.shuffle(indices)
        shown = 0
        for idx in indices:
            if shown >= VISUALIZE_SAMPLES:
                break
            show_match(q_sample_paths[idx], g_sample_paths[idx], 
                      y_pred[idx], y_true[idx], sim_scores[idx])
            shown += 1

    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📊 Final Results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"📁 Detailed results saved in: {csv_filename}")

# --- RUN ---
if __name__ == "__main__":

# ---------------- Entry Point ---------------- #
    # Ask for test/val path
    print(f"Task_A Validation")
    image_folder = input("🔍 Enter path to validation or test folder: ").strip()

    # Base directory of the script
    BASE_DIR = Path(__file__).resolve().parent

    # Point to model files relative to current folder
    MODEL_1_PATH = BASE_DIR / "Task_A" / "resnet18_M1_state_dictionary.pth"
    MODEL_2_PATH = BASE_DIR / "Task_A" / "resnet18_M2_state_dictionary.pth"

    model_paths = [MODEL_1_PATH, MODEL_2_PATH]
    model_weights = [0.6, 0.4]

    run_ensemble_inference(model_paths, model_weights, image_folder)
    try:
        run_val_inference()
    except KeyboardInterrupt:
        print("\n\n⏹️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()