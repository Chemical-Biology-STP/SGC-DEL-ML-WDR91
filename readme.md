The trained models are saved in `models` and named after the ensemble size and 
fingerprints used. The model used for initial screening was `WDR91_bECFP6_E5.pkl`.
The model used for rescreening to better assess prediction confidence was `WDR91_bECFP6_ECFP6_AtomPair_E5.pkl`

The model code is in `train.py` and implements a few fingerprinting functions and the model class. You'll need this full file to load the pickle. Not ideal, but it was how I built it 
back I learned about good best practices for ML. 

To use the model for screening SMILES, you can use the `screen_smiles` function which takes a list of SMILES and returns preds to you. Or you can use `screen` which takes the path to a `.smi` file and screens all SMILES in the file and saves them.

Prediction include both a "PRED" which is the probability a compound is a binder and a "CONF" which is the models own assessment of how trustworthy this specific prediction is. While these might sound similar, CONF is based off of entropy in prediction. It is very possible to get a PRED of 0.80 but a conf a 0.20, meaining while it thinks there is a good chance its active, it also observed the prediction to have high entopy, meaning that small changes in training data resulted in major changes in the PRED. This is evidence that the prediction could be a false positive.