# nn-molecular-odor-prediction

# Molecular Odor Prediction  
Predict the chances of **three key odors (Fruity, Woody, Sulfurous)** from molecular **SMILES** strings using a trained neural network.

---

## 🌍 Try It Live!  
🚀 **Deployed Web App:** [Molecular Odor Prediction](https://molecular-odor-prediction.streamlit.app/)  

---

## 📂 Dataset  
- Source: [Multi-Labelled SMILES Odors Dataset (Kaggle)](https://www.kaggle.com/datasets/aryanamitbarsainyan/multi-labelled-smiles-odors-dataset)  
- Contains molecular **SMILES strings** labeled with multiple **odor categories**.  
- **Three odors selected for prediction:**  
  - **Fruity**  
  - **Woody**  
  - **Sulfurous**  

---

## ⚙️ How It Works  
1. **SMILES to Fingerprint Conversion**  
   - Uses **RDKit** to generate **1024-bit Morgan fingerprints**.  
2. **Odor Probability Prediction**  
   - A **Neural Network (PyTorch)** predicts odor probabilities.  
3. **Streamlit Web App**  
   - Allows **real-time odor prediction** from user-input SMILES.  

---

## 📦 Model & Training  

### 🔹 Model Architecture  
- **Input:** 1024-bit Morgan Fingerprint  
- **Hidden Layers:** Fully connected layers with **BatchNorm, ReLU, and Dropout**  
- **Output:** 3 neurons (one per odor), **Sigmoid activation** for probability output.  

### 🔹 Training Details  
- **Loss Function:** `BCELoss()` (Binary Cross Entropy)  
- **Optimizer:** `Adam (lr=0.001)`  
- **Dataset Split:** 80% train, 20% test  
- **Early Stopping & Learning Rate Scheduling** applied  

---

## 💻 Streamlit Web App  

### 🌐 Features  
✔ **User inputs SMILES string**  
✔ **Predicts chances of Fruity, Woody, Sulfurous odors**  
✔ **Progress bars for better visualization**  

--- 

## Example Predictions  
Here are some real examples tested with the model:  

### Example 1: `CC(C)(C)C1CCC2(CCC(=O)O2)CC1`  
- **Ground Truth:** `{Fruity: 0, Woody: 1, Sulfurous: 0}`  
- **Model Prediction:**  
  - **Fruity:** 1.72%  
  - **Woody:** 99.57%  
  - **Sulfurous:** 0.89%  
- ✅ The model correctly predicts it as **WOODY**!  

---

### Example 2: `CCCCCCCCCCCC(=O)OCC`  
- **Ground Truth:** `{Fruity: 1, Woody: 0, Sulfurous: 0}`  
- **Model Prediction:**  
  - **Fruity:** 99.26%  
  - **Woody:** 1.73%  
  - **Sulfurous:** 2.47%  
- ✅ The model correctly predicts it as **FRUITY**!  

---

### Example 3: Try it yourself!  
Curious to see how the model performs? Try the following **SMILES string** in the web app:  

`CC=CC(=O)OC(CC)C(C)C`

---

## About This Project  

I have built this project from scratch, handling every aspect from **data preprocessing** to **model development** and **deployment**. Initially, I experimented with **traditional machine learning algorithms** such as Random Forest and XGBoost. However, after thorough evaluation, I found that **neural networks** provided significantly better performance for odor prediction.  

To make the model accessible, I developed a **Streamlit web application** and personally **deployed it** on Streamlit Cloud. This project demonstrates my ability to work through the entire machine learning pipeline, from **data processing to model optimization and deployment**.  
