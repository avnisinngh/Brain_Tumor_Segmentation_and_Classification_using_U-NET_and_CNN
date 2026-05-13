# 🧠 Brain Tumor Segmentation & Classification using U-Net + CNN

A deep learning pipeline that combines **U-Net** for tumor segmentation and a **CNN** for tumor classification on brain MRI scans. The system first localizes the tumor region, then classifies it into one of four categories.

---

## 📌 Project Overview

| Task | Model | Accuracy |
|------|-------|----------|
| Tumor Segmentation | U-Net | ~99.09% pixel accuracy, F1: 0.718 |
| Tumor Classification | CNN | ~90.5% |
| Full Pipeline Accuracy | U-Net+CNN | ~91.08% |

The full pipeline:
1. **U-Net** segments the tumor region from an MRI scan
2. The largest ROI (Region of Interest) is cropped from the mask
3. **CNN** classifies the cropped region into a tumor type

---

## 🗂️ Dataset

The dataset is organized into two main tasks:

```
DATASET/
├── classification/
│   ├── Training/
│   │   ├── glioma/        (1321 images)
│   │   ├── meningioma/    (1339 images)
│   │   ├── pituitary/     (1457 images)
│   │   └── notumor/       (1595 images)
│   └── Testing/
│       ├── glioma/        (300 images)
│       ├── meningioma/    (306 images)
│       ├── pituitary/     (300 images)
│       └── notumor/       (405 images)
└── Segmentation/
    ├── Glioma/            (1108 images + masks)
    ├── Meningioma/        (1416 images + masks)
    └── Pituitary tumor/   (1860 images + masks)
```

**Total training samples:** 5,712 | **Total test samples:** 1,311  
**Segmentation train/test split:** 1,863 / 329 (85/15)

---

## 🏗️ Model Architecture

### U-Net (Segmentation)

```
Input (256×256×1)
    ↓
Encoder:
  Conv2D(32) → Conv2D(32) → MaxPool
  Conv2D(64) → Conv2D(64) → MaxPool
  Conv2D(128) → Conv2D(128) → MaxPool
  Conv2D(256) → Conv2D(256) → MaxPool
    ↓
Bottleneck:
  Conv2D(512) → Conv2D(512)
    ↓
Decoder (with skip connections):
  ConvTranspose(256) + skip → Conv2D(256) × 2
  ConvTranspose(128) + skip → Conv2D(128) × 2
  ConvTranspose(64)  + skip → Conv2D(64) × 2
  ConvTranspose(32)  + skip → Conv2D(32) × 2
    ↓
Output: Conv2D(1, sigmoid) → Binary Mask
```

**Key design choices:**
- Skip connections retain tumor boundary details
- Deep bottleneck (512 filters) enables strong feature extraction
- Sigmoid activation for binary mask output

### CNN (Classification)

```
Input (128×128×1)
  → Conv2D(32) + MaxPool
  → Conv2D(64) + MaxPool
  → Conv2D(128) + MaxPool
  → Flatten
  → Dense(128, relu)
  → Dense(4, softmax)
```

---

## ⚙️ Training Configuration

| Parameter | U-Net | CNN |
|-----------|-------|-----|
| Image Size | 256×256 | 128×128 |
| Batch Size | 16 | 32 |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss | Binary Crossentropy | Categorical Crossentropy |
| Epochs | 45 | 40 |
| Channels | Grayscale (1) | Grayscale (1) |

### Data Augmentation (applied during training)
- Random horizontal & vertical flips
- Random 90° rotations
- Random brightness adjustment (U-Net: none; CNN: ±0.2)
- Random contrast adjustment (CNN only: 0.8–1.2)

---

## 📊 Results

### U-Net Segmentation
- **Pixel Accuracy:** 99.09%
- **F1 Score:** 0.7175
- **Confusion Matrix (pixel-level):**

```
               Predicted Background   Predicted Tumor
Actual Background      21,117,596          72,312
Actual Tumor              123,166         248,270
```

### CNN Classification
- **Test Accuracy:** 90.54%
- **Confusion Matrix:**

|            | glioma | meningioma | pituitary | notumor |
|------------|--------|------------|-----------|---------|
| glioma     | 255    | 41         | 0         | 4       |
| meningioma | 12     | 234        | 13        | 47      |
| pituitary  | 1      | 3          | 294       | 2       |
| notumor    | 1      | 0          | 0         | 404     |

Pituitary and notumor classes achieved the highest per-class accuracy. Meningioma showed the most confusion with other classes.

### 🔗 Full Pipeline (U-Net + CNN combined)

The complete pipeline chains U-Net segmentation → ROI crop → CNN classification, evaluated across all 1,311 test images.

- **Full Pipeline Accuracy: 91.08%**

**Sample predictions (from `test_random()`):**

| Test Case | Actual | Predicted | Confidence | Result |
|-----------|--------|-----------|------------|--------|
| Sample 1 | meningioma | meningioma | 79.45% | ✅ |
| Sample 2 | meningioma | meningioma | 68.91% | ✅ |

**Summary of all model results:**

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| U-Net | Segmentation | Pixel Accuracy | 99.09% |
| U-Net | Segmentation | F1 Score (pixel) | 0.7175 |
| CNN (standalone) | Classification | Test Accuracy | 90.54% |
| **U-Net + CNN (full pipeline)** | **Segmentation + Classification** | **Test Accuracy** | **91.08%** |

> The full pipeline (91.08%) outperforms the standalone CNN (90.54%), confirming that U-Net segmentation and ROI cropping provides a meaningful boost to classification accuracy.

---

## 🔁 Full Inference Pipeline

```python
def full_pipeline_optimized(img_path, true_label=None):
    # Step 1: Load grayscale MRI image
    # Step 2: Preprocess for U-Net (resize to 256×256, normalize)
    # Step 3: Predict segmentation mask (threshold > 0.40)
    # Step 4: Apply mask to image
    # Step 5: Crop largest tumor ROI using bounding box
    # Step 6: Resize ROI to 128×128 for CNN input
    # Step 7: Predict tumor class + confidence
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/avnisinngh/Brain_Tumor_Segmentation_and_Classification_using_U-NET_and_CNN.git
cd Brain_Tumor_Segmentation_and_Classification_using_U-NET_and_CNN
```

2. Upload the dataset zip to `/content/` in Google Colab

3. Open `unet&cnn_(1).ipynb` in Google Colab and run all cells

### Using Pretrained Models

```python
import tensorflow as tf

unet_model = tf.keras.models.load_model("unet_final.h5", compile=False)
cnn_model  = tf.keras.models.load_model("cnn_final.h5", compile=False)

# Run full pipeline
full_pipeline_optimized("path/to/mri.jpg", true_label="glioma")
```

---

## 📁 Repository Structure

```
├── unet&cnn_(1).ipynb      # Main notebook
├── unet_final.h5           # Trained U-Net model
├── cnn_final.h5            # Trained CNN model
└── README.md
```

---

## 🔬 Classes

| Label | Description |
|-------|-------------|
| `glioma` | Tumor arising from glial cells |
| `meningioma` | Tumor of the meninges (brain lining) |
| `pituitary` | Tumor of the pituitary gland |
| `notumor` | Healthy brain scan (no tumor) |

---

## 🛠️ Tech Stack

- **Framework:** TensorFlow / Keras
- **Data handling:** Pandas, NumPy
- **Image processing:** OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** Scikit-learn
- **Environment:** Google Colab

---

## 📈 Future Improvements

- Add Dice Loss for better segmentation training
- Use transfer learning (e.g., EfficientNet, ResNet) for classification
- Integrate Grad-CAM for explainability
- Deploy as a web app using Flask or Streamlit
- Extend to 3D MRI volumes

---

## 👤 Author

**Avni Singh**  
GitHub: [@avnisinngh](https://github.com/avnisinngh)
