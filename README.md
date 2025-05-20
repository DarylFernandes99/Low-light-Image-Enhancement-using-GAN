# Low-Light Image Enhancement using GAN

A deep learning project that transforms images taken in poor lighting conditions into well-illuminated, enhanced versions using Generative Adversarial Networks (GANs).

## 🌓 Overview

This project tackles the challenge of enhancing low-light images through advanced deep learning techniques. The system uses a custom GAN architecture to generate high-quality, well-illuminated images from dark, poorly lit input images - as if they were taken under optimal lighting conditions.

![Image Enhancement Example](./Images/Discriminator.JPG)

## ✨ Features

- Transforms dark, low-light images into clear, well-illuminated versions
- Implements a sophisticated encoder-decoder GAN architecture
- Preserves image details while enhancing brightness and contrast
- Handles various types of low-light conditions (night scenes, indoor low lighting, etc.)
- Supports multiple image formats (JPG, PNG)

## 🧠 GAN Architecture

The project implements a custom GAN with three main components:

### Generator
Split into two parts:
- **Encoder**: Extracts features from the low-light input image
  ![Encoder Architecture](./Images/Encoder.JPG)
  
- **Decoder**: Reconstructs an enhanced version of the image
  ![Decoder Architecture](./Images/Decoder.JPG)

### Discriminator
Distinguishes between real well-lit images and generated enhanced images
![Discriminator Architecture](./Images/Discriminator.JPG)

## 📊 Datasets

The model was trained on multiple datasets:
- **LOL Dataset**: Low Light Paired Dataset ([Link](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view))
- **Synthetic Image Pairs**: Created from raw images ([Link](https://drive.google.com/file/d/1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F/view))
- **SID Dataset**: Sony ([Link](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)) and Fuji ([Link](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip)) low-light images
- **SICE Dataset**: Single Image Contrast Enhancement - [Part1](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view) | [Part2](https://drive.google.com/file/d/16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN/view)
- **Custom Dataset**: Created by adding noise to Google scraped images ([Kaggle Link](https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn))

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.4.1+
- Keras
- NumPy
- OpenCV
- PIL (Python Imaging Library)

## 💻 Development Environment

- **GPU**: Nvidia Tesla T4 16GB / Nvidia Tesla P100 16GB
- **RAM**: 12GB
- **Software**: Python 3.7, TensorFlow 2.4.1

## 🚀 Getting Started

### 1. Data Preparation (png2npz.py)

Convert your images from PNG/JPEG/JPG to NPZ format:

```python
# Set up your directory structure:
# path/
# ├── ground_truth/  # Well-lit images
# └── low/           # Low-light images

# Edit png2npz.py to set the correct path
path = "your_dataset_path/"

# Run the conversion script
python png2npz.py

# The output will be saved as dataset.npz
```

### 2. Model Training (Low_Light_Image_Enhancement_using_GAN.ipynb)

This notebook is designed to run on Google Colab with GPU acceleration:

1. Upload the NPZ dataset to Google Drive
2. Open the notebook in Colab and mount your Drive
3. Update the path variables to point to your dataset
4. Set the batch size (default is 12)
5. Execute all cells in sequence
6. The model will be saved at regular intervals

### 3. Image Enhancement (main.py)

Use the trained model to enhance your low-light images:

```python
# Edit main.py to set your paths
model_path = "path_to_saved_model"
image_path = "path_to_low_light_image"

# Run the script
python main.py

# The enhanced image will be saved as output.png
```

## 📝 Research Publication

This project has been published in a research paper:
- [IRJET Publication](https://www.irjet.net/archives/V8/i6/IRJET-V8I6136.pdf)

## 📄 License

This project is licensed under the terms of the included LICENSE file.

## 🌟 Results

The GAN model achieves significant improvements in:
- Overall brightness
- Contrast enhancement
- Detail preservation
- Color accuracy
- Noise reduction

## 🔄 Workflow

1. **Data Preparation**: Images are paired (low-light and ground truth)
2. **Preprocessing**: Images are resized to 256×256 and normalized
3. **Training**: The GAN learns to transform low-light to enhanced images
4. **Inference**: New low-light images are processed through the generator
5. **Post-processing**: Median blur is applied to reduce any artifacts
