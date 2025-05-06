🧠 Tumor Cancer Detection

A deep learning-based project to automatically detect and classify brain tumors from MRI images using computer vision and machine learning techniques.

📂 Dataset
It contains four classes:

glioma

meningioma

pituitary

no tumor

Images are in JPG format and have been resized to 128x128 pixels with 3 color channels (RGB).

🧪 Project Pipeline
Preprocessing

Image resizing to 128×128

CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement

Noise reduction

Morphological operations for feature enhancement

Segmentation

Region Growing Algorithm to segment the tumor area.

Feature Extraction

LBP (Local Binary Patterns) for texture

Morphological features: area, perimeter, compactness...

Classification

A pre-trained ResNet50-based custom CNN or an SVM classifier used for final classification.

🖥️ Requirements
Python 3.8+

PyTorch

OpenCV

NumPy

matplotlib

albumentations (for augmentation)
