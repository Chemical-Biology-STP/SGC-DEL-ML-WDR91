import pickle

import pandas as pd
import streamlit as st

from scripts.train import Model  # Assuming Model is defined in train.py


# Function to load the trained model
def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


# Function to run predictions
def run_predictions(model, smiles_list):
    """Use the loaded model to predict activities for a list of SMILES."""
    preds, confs = Model.screen_smiles(model, smiles_list)
    results = pd.DataFrame(
        {
            "SMILES": smiles_list,
            "PREDICTION": [round(p, 4) for p in preds],
            "CONFIDENCE": [round(c, 4) for c in confs],
        }
    )
    return results


# Streamlit app
def main():
    st.title("SMILES Prediction App")

    # Reference to the ChemRxiv article with DOI link
    st.markdown(
        """
    This app is based on the approach discussed in the article 
    **[Enabling Open Machine Learning of DNA Encoded Library Selections to Accelerate the Discovery of Small Molecule Protein Binders](https://doi.org/10.26434/chemrxiv-2024-xd385)**, 
    published on *ChemRxiv*. The study focuses on using machine learning to discover binders for the WDR91 protein.
    """
    )

    st.markdown(
        """
    **Explanation of Output:**

    - **PREDICTION**: This value represents the probability that a compound is a binder (i.e., an active compound).
    - **CONFIDENCE**: This value represents the model's assessment of how reliable this specific prediction is, based on prediction entropy.
    
    **Important Caveat**: It is possible to have a high PREDICTION (e.g., 0.80) but a low CONFIDENCE (e.g., 0.20). This indicates that the model sees the compound as likely active but with high uncertainty. Small changes in training data may significantly alter the PREDICTION, suggesting the prediction could be a false positive.
    """
    )

    # Model loading
    model_path = "models/WDR91_bECFP6_E5.pkl"
    model = load_model(model_path)

    # File upload section
    st.header("Upload your SMILES file (.smi or .txt)")
    smiles_file = st.file_uploader("Choose a file", type=["smi", "txt"])

    if smiles_file:
        # Reading the SMILES strings from the file
        smiles_list = smiles_file.read().decode("utf-8").splitlines()
        smiles_list = [smi.strip() for smi in smiles_list if smi.strip()]

        if len(smiles_list) > 0:
            # Run predictions and display the results
            results = run_predictions(model, smiles_list)
            st.subheader("Prediction Results")
            st.dataframe(results)

            # Option to download results as a TSV file
            tsv_data = results.to_csv(sep="\t", index=False).encode("utf-8")
            st.download_button(
                label="Download results as TSV",
                data=tsv_data,
                file_name="predictions.tsv",
                mime="text/tsv",
            )


if __name__ == "__main__":
    main()
