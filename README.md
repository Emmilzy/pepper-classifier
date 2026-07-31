# 🌶️ Pepper Leaf Disease Classifier — Healthy vs Bacterial Spot

GET 324: Cloud Computing and AI Model Deployment for Engineering Applications
Laboratory Exercise 10 (Mini-Project)

## Project Overview

This project implements a binary image classifier that distinguishes between:

- **Healthy** bell pepper leaves
- **Bacterial Spot** infected bell pepper leaves

The model is built using **transfer learning (MobileNetV2, pretrained on ImageNet)**
with TensorFlow/Keras, and is deployed as an interactive **Streamlit** web
application that lets a user upload a leaf image and receive an instant
prediction with a confidence score.

## Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes used:**
  - `Pepper__bell___healthy`
  - `Pepper__bell___Bacterial_spot`
- Only these two class folders were used to build a binary classification
  dataset (all other crop/disease folders in PlantVillage were excluded).

## Project Structure

```
pepper-classifier/
├── app.py                 # Streamlit application
├── train_model.py         # Model training script (run in Colab or locally)
├── pepper_model.keras     # Saved trained model (generated after training)
├── requirements.txt       # Python dependencies
├── README.md               # Project documentation
└── report.md               # Project report
```

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/pepper-classifier.git
   cd pepper-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (If `pepper_model.keras` is not already present) train the model:
   - Download the PlantVillage dataset, keep only the two pepper folders
     inside a directory named `dataset/`, then run:
   ```bash
   python train_model.py
   ```

4. Launch the app:
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`)
   and upload a pepper leaf image to classify it.

## How to Use the Application

1. Open the deployed app link (see below).
2. Click **"Browse files"** and upload a `.jpg`, `.jpeg`, or `.png` image of
   a bell pepper leaf.
3. Click **Classify**.
4. The app displays the predicted class (**Healthy** or **Bacterial Spot**)
   along with a confidence percentage.

## Model Details

- **Architecture:** MobileNetV2 (frozen base, then fine-tuned) + Global
  Average Pooling + Dropout + Dense(1, sigmoid)
- **Input size:** 224 × 224 × 3
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam (1e-3 for feature extraction, 1e-5 for fine-tuning)
- **Data augmentation:** random flip, rotation, zoom, contrast

## Deployment

The application is deployed on **Streamlit Community Cloud**.

🔗 **Live App:** https://pepper-classifier-buxjp8thepzegeuucgszvk.streamlit.app

To deploy your own copy:
1. Push this repository to GitHub (include `pepper_model.keras`, or use Git
   LFS / an external download link if the file is large).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and create a new app pointing to `app.py` in this repository.
3. Wait for the build to complete and share the generated URL.

## Team Members

| Name                  | Registration Number | GitHub Username |
|AKPAN,EMMANUEL NSEMEKE|22/EG/EE/2055 | Emmilzy |
|DICK,MBEREOBONG AKPAN|22/EG/EE/2075|GONZALEZDICK|
| UBAK, GORDON INI-OBONG| 22/EG/EE/2045 | GordonUbak |
| Sunday,Nsikakabasi Lawrence| 22/EG/EE/2005 | nsikaksunday621-ship-it |
| Akpan, Saviour Friday| 22/EG/EE/2025| Sahviour205-web| 
|                       |              |                         |
|                        |              |                        |

## Course Learning Outcomes Addressed

- **CLO5:** Designed, trained, and evaluated a CNN (transfer learning) model
  for image classification using TensorFlow/Keras.
 - **CLO7:** Deployed the trained model as a cloud-based web application using
  Streamlit, managed via Git/GitHub.
- **CLO8:** Documented the experimental process and results in this README
  and the accompanying report.
