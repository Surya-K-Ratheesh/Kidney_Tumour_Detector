# 🩺 KidneyInsight: AI Tumor Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **KidneyInsight** is an AI-powered diagnostic web application designed to detect the presence of tumors from Kidney CT Scan images using deep learning.

---

## 📖 Overview

KidneyInsight leverages a custom-trained Convolutional Neural Network (CNN) to analyze Kidney CT Scans and classify them as Normal or Tumor. Built with an intuitive **Streamlit** front-end, the application provides doctors and users with real-time predictions and visual explainability using **Grad-CAM**, highlighting the exact region of the scan that influenced the model's decision.

## ✨ Features

- **Upload & Analyze**: Easily upload Kidney CT Scans (JPG, JPEG, PNG).
- **Deep Learning Powered**: Utilizes a highly accurate, pre-trained TensorFlow/Keras H5 model for prediction.
- **Explainable AI (XAI)**: Generates a **Grad-CAM Heatmap** to localize the tumor and extract a bounding box so users aren't left guessing.
- **Fast & Interactive UI**: Beautiful and responsive visualization powered by Streamlit.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, Pillow
- **Data Manipulation**: NumPy
- **Explainability**: Grad-CAM

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You also need the following libraries:

- `streamlit`
- `tensorflow`
- `numpy`
- `opencv-python`
- `Pillow`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Surya-K-Ratheesh/Kidney_Tumour_Detector.git
   cd Kidney_Tumour_Detector
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   *(If a `requirements.txt` is added later, run `pip install -r requirements.txt`. Otherwise, install manually:)*
   ```bash
   pip install streamlit tensorflow numpy opencv-python Pillow
   ```

### Running the Application

1. Ensure your trained model `best_model.h5` is located in the root directory.
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided Local URL (usually `http://localhost:8501`) in your web browser.

---

## 📸 Usage

1. Open the KidneyInsight Web App.
2. Drag and drop or browse to upload a Kidney CT Scan image.
3. Click the **Analyze Scan** button.
4. View the result:
   - **No Tumor Detected:** Model confidently predicts the kidney is normal.
   - **Tumor Detected:** The app displays the confidence score, plots a Grad-CAM overlay to highlight the affected area, and provides bounding box coordinates.

---

## 📂 Project Structure

```text
Kidney_Tumour_Detector/
│
├── app.py                   # Main Streamlit web application
├── analyze_image.py         # Script for individual image analysis
├── show_b64.py              # Script for Base64 image rendering
├── best_model.h5            # Pre-trained deep learning model
├── src/                     # Source folder containing modules
│   ├── explainability/      # Grad-CAM and visualization scripts
│   ├── model/               # Model utilities
│   ├── preprocessing/       # Image preprocessing scripts
│   └── utils/               # General utility functions
├── exported_images/         # Directory for exported results
└── processed_clahe/         # pre-processed datasets/images
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Surya-K-Ratheesh/Kidney_Tumour_Detector/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Disclaimer: KidneyInsight is an educational and supportive tool. It must not be strictly used as a standalone diagnostic medical instrument. Always consult a certified radiologist or doctor.*
