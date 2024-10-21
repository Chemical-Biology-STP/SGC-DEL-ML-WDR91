
# SGC-DEL-ML-WDR91

This Streamlit-based web app predicts activities for chemical compounds represented by SMILES strings. It is designed to help accelerate the discovery of small molecule binders for the **WDR91 protein**, using machine learning models trained on DNA-Encoded Library (DEL) selections.

## Features

- Upload a file containing SMILES strings and get predictions for each compound.
- Displays **PREDICTION** and **CONFIDENCE** values for each compound.
- Download the prediction results as a **TSV** file.

## Models

The trained models are stored in the `models` folder and are named based on the ensemble size and fingerprints used for training. Two models were used for different stages of screening:

1. **Initial Screening Model**: `WDR91_bECFP6_E5.pkl`
   - Used for the initial screening of SMILES strings.
2. **Rescreening Model**: `WDR91_bECFP6_ECFP6_AtomPair_E5.pkl`
   - Used for rescreening to better assess prediction confidence.

### Prediction Details

- **PRED**: The probability that a compound is a binder (i.e., an active compound).
- **CONF**: The model's assessment of how trustworthy the prediction is, based on prediction entropy. A high **PRED** with a low **CONF** indicates high uncertainty in the prediction and suggests it could be a false positive.

## Model Code

The model code is located in `train.py` and implements several fingerprinting functions and the model class. You'll need the full `train.py` file to load the model pickle files. This is not an ideal setup, but it reflects earlier practices before implementing better machine learning standards.

### Screening Functions

To use the model for screening SMILES strings, you have two options:

1. **`screen_smiles` function**:
   - Takes a list of SMILES strings and returns predictions (`PRED`) and confidence values (`CONF`).

2. **`screen` function**:
   - Takes the path to a `.smi` file, screens all SMILES in the file, and saves the results.

## Reference

This app is based on the approach discussed in the ChemRxiv article:

**[Enabling Open Machine Learning of DNA Encoded Library Selections to Accelerate the Discovery of Small Molecule Protein Binders](https://doi.org/10.26434/chemrxiv-2024-xd385)**.

## Prerequisites

Before running the app, you need to install some software:

1. **Anaconda/Miniconda**: A package manager that helps manage dependencies easily.

   - Download and install Anaconda or Miniconda from here: [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

## Getting Started

### 1. Clone or download the project

First, download the project to your local machine. If you're using Git, you can clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

If you have downloaded the zip file, unzip it and navigate to the project folder.

### 2. Set up the Conda environment

The project includes a file called `environment.yaml` that lists the necessary dependencies. You will use this file to create a Conda environment.

Open your terminal or Anaconda Prompt and navigate to the directory where the `environment.yaml` file is located.

Run the following command to create a new Conda environment:

```bash
conda env create -f environment.yaml
```

This will create an environment named `SGC_DEL_ML_WDR91` with all required dependencies.

### 3. Activate the Conda environment

Once the environment is created, activate it by running:

```bash
conda activate SGC_DEL_ML_WDR91
```

### 4. Running the Streamlit App

After setting up the environment, you can run the app using the following command:

```bash
streamlit run app.py
```

This will start a Streamlit server and open the app in your default web browser.

### 5. Upload Your SMILES File

Once the app is running, you can upload a **.smi** or **.txt** file containing SMILES strings (one per line). The app will predict the activities of the compounds and display the results.

### Supported File Formats

- **SMI**: A file containing SMILES strings, with one molecule per line.
- **TXT**: A simple text file containing SMILES strings, one per line.

## Troubleshooting

1. **Environment issues**: If you encounter issues during environment setup, ensure that you have installed Anaconda or Miniconda correctly and that the `environment.yaml` file matches the one in the project.
2. **Package installation**: If some dependencies are missing, ensure that you are running the command within the activated environment: `conda activate SGC_DEL_ML_WDR91`.
3. **App not launching**: Ensure that you're running `streamlit run app.py` from the correct directory and that the environment is activated.

## Raising Issues

If you encounter any problems or have suggestions, please raise an issue in the GitHub repository. We will address your concerns as soon as possible.

## License

This project is licensed under the MIT License.

---

This README provides everything you need to get started with **SGC-DEL-ML-WDR91**. If you encounter any issues, don't hesitate to raise an issue via the GitHub repository!
