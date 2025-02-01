import streamlit as st
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import OdorNN  # Import the defined model

# Load the trained model
model_path = "odor_model.pth"
model = OdorNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Convert SMILES to Fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None  # Return None if invalid SMILES

# Predict Odor
def predict_odor(smiles):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return "Error: Invalid SMILES input."
    
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        predicted_probs = model(fingerprint_tensor).cpu().numpy().flatten()  # Get probabilities

    # Odor Labels
    odors = ["Fruity", "Woody", "Sulfurous"]

    # Format predictions
    predicted_probs_dict = {odors[i]: round(predicted_probs[i] * 100, 2) for i in range(len(odors))}
    return predicted_probs_dict

st.set_page_config(page_title="Molecular Odor Prediction", page_icon="🔬")

# Streamlit UI
st.title("Molecular Odor Prediction (Fruity, Woody, and Sulfurous Odors)")
st.write("Enter a SMILES string to predict the probability of different odor categories.")

smiles = st.text_input("Enter SMILES:")

if st.button("Predict"):
    if smiles:
        result = predict_odor(smiles)
        if isinstance(result, str):
            st.error(result)  # Display error if invalid SMILES
        else:
            st.success("Prediction Successful")

            # Show predictions in a structured way
            st.subheader("Predicted Odor Probabilities")

            # Progress Bars for visual representation
            for odor, prob in result.items():
                st.write(f"**{odor}**: {prob}%") 
                st.progress(int(prob))  # Show as progress bar
                
    else:
        st.warning("Please enter a valid SMILES string.")

st.markdown(
    """
    <hr style="border:1px solid #ddd;">
    <p style="text-align:center;">
        Made by <a href="https://github.com/akshatrksingh" target="_blank"><strong>Akshat Singh</strong></a>
    </p>
    """,
    unsafe_allow_html=True
)