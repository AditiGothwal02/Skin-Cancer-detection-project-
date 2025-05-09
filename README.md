
# 🩺 Skin Cancer Detection Using CNN

Skin cancer is one of the most common types of cancer, accounting for millions of diagnoses each year worldwide. It is caused by the uncontrolled growth of abnormal skin cells, often due to prolonged exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds. Skin cancer can be classified into three primary types: basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and melanoma, with melanoma being the most aggressive and potentially fatal form if not detected early. Early detection significantly increases the chances of successful treatment, yet access to dermatologists and diagnostic tools is often limited, especially in remote or underserved areas. To address this issue, advancements in machine learning and deep learning have made it possible to automate skin cancer detection with high accuracy, helping healthcare providers deliver faster and more reliable diagnoses.

---
# Purpose

This project aims to develop an AI-driven system for automated skin cancer detection using Convolutional Neural Networks (CNNs). The system is designed to classify skin lesions as benign (non-cancerous) or malignant (cancerous) based on images uploaded by the user. The project begins with understanding the requirements and defining the scope clearly, ensuring a solid foundation for what needs to be achieved. Technologies such as TensorFlow/Keras for building the CNN model, for image preprocessing (resizing, normalization), and Python for scripting are utilized throughout the development process. The team then enters a phase of breaking down the project into specific tasks, creating a to-do list, and setting achievable goals. The project progresses with regular feedback from mentors, adjustments to improve performance, and the use of Stream lit or Flask to create a user-friendly web interface for deployment. Following development, testing, and debugging, the final stage involves presenting the polished system, showcasing its functionality, addressing challenges, and demonstrating the technological stack used in the project.

---

## ⚙️ System Requirements

- Python 3.8 or later
- pip
- virtualenv
- Streamlit
- TensorFlow

---

## 🚀 Steps to Replicate

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/skin-cancer-detection-cnn.git
cd skin-cancer-detection-cnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:
```
streamlit
tensorflow
numpy
pillow
opencv-python
scikit-learn
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 🔒 Image Validation

This app automatically rejects:
- Images that are not skin lesions
- Empty or invalid uploads

---

## 📷 Prediction Output

The app predicts cancer type with a message like:
> "This is Melanoma cancer"

---
