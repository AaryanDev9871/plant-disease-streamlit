# 🌿 Plant Disease Detection Using Deep Learning

A deep learning web application that predicts plant diseases from leaf images using a trained **MobileNetV2-based image classification model**. The application is built with Streamlit and deployed on Streamlit Community Cloud.

## 🚀 Live Demo

**Try the application:** https://plant-disease-detector-2998.streamlit.app/

## 📌 Project Overview

Plant diseases can affect crop health, productivity, and agricultural yield. Identifying potential diseases early can help farmers and gardeners make more informed decisions about plant care.

This project provides a simple web interface where users can upload a plant leaf image and receive a predicted disease class from a trained deep learning model.

The application demonstrates how deep learning and web technologies can be combined to build an accessible image classification tool.

## ✨ Features

- **Leaf Image Upload:** Upload a plant leaf image for analysis.
- **Deep Learning Predictions:** Classify uploaded images using a trained MobileNetV2-based model.
- **38 Output Classes:** Predict among the model's 38 configured plant disease and healthy-leaf classes.
- **Interactive Web Interface:** Use the application through a Streamlit interface.
- **Image Preprocessing:** Prepare uploaded images for the model's expected input dimensions.
- **Cloud Deployment:** Access the application online without installing it locally.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model loading and inference |
| MobileNetV2 | Base architecture for image classification |
| Streamlit | Interactive web application |
| NumPy | Numerical array operations |
| Pillow (PIL) | Image loading and preprocessing |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

## 🧠 Model Details

- **Model architecture:** MobileNetV2-based convolutional neural network
- **Task:** Multiclass image classification
- **Input image size:** 224 × 224 pixels
- **Output:** 38 class scores, represented as probabilities by the model's softmax output layer
- **Model file:** `plant_disease_model.keras`
- **Class mapping:** `class_names.json`

The model predicts the class most strongly supported by its output scores. Prediction confidence does not guarantee that the predicted disease is correct, particularly for images that differ from the training data.

## 📂 Project Structure

```text
plant-disease-streamlit/
│
├── app.py
├── plant_disease_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application and prediction logic.
- `plant_disease_model.keras` — Saved deep learning model.
- `class_names.json` — Class labels used to interpret model outputs.
- `requirements.txt` — Python package dependencies.
- `README.md` — Project documentation.

## ⚙️ Run the Project Locally

### Prerequisites

Install Python and Git on your computer.

### 1. Clone the repository

```bash
git clone https://github.com/AaryanDev9871/plant-disease-streamlit.git
```

### 2. Open the project directory

```bash
cd plant-disease-streamlit
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run the environment's Python directly using `.\venv\Scripts\python.exe`.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser and upload a leaf image.

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

To deploy your own copy:

1. Push the project files to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and select your repository and branch.
4. Set `app.py` as the main application file.
5. Deploy the application.

Make sure the model file, class mapping, and dependency configuration are available to the deployed app.

## 📸 How to Use

1. Open the [live application](https://plant-disease-detector-2998.streamlit.app/).
2. Upload an image containing a plant leaf.
3. Wait for the model to process the image.
4. Review the predicted class and any additional results displayed by the application.

For better results, use a clear, well-lit image with the leaf visible and in focus.

## ⚠️ Limitations

- Prediction quality depends on the training data and the similarity of uploaded images to that data.
- Images downloaded from the internet may have different backgrounds, lighting, angles, and symptoms, which can reduce prediction reliability.
- A high model confidence score does not necessarily mean a prediction is correct.
- The application is an educational project and should not be treated as a definitive plant disease diagnosis or a substitute for expert agricultural advice.

## 🔮 Future Improvements

- Display the top three predicted classes and their confidence scores.
- Improve performance on real-world images from different environments.
- Evaluate the model using a separate set of labelled internet or field images.
- Add clearer warnings for uncertain predictions.
- Improve the interface and provide more informative prediction results.
- Add model evaluation metrics and a confusion matrix to the documentation.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Deep learning and image classification
- TensorFlow and Keras model inference
- Image preprocessing
- Building interactive applications with Streamlit
- Managing source code with Git and GitHub
- Deploying a Python machine learning application to the cloud

## 👨‍💻 Author

**Aaryan Dev Shridhar**

B.Tech in Computer Science and Engineering

GitHub: [AaryanDev9871](https://github.com/AaryanDev9871)

---

*This project was developed for learning and demonstration purposes. Predictions should be independently verified before making plant treatment decisions.*
