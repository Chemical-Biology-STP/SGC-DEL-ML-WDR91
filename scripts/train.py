import abc
import inspect
import os
from copy import deepcopy
from functools import partial
from time import time
from typing import Dict, Union, Optional
import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

#rdkit imports
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, MolFromSmiles
from rdkit.Chem import RDKFingerprint
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, \
    average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm import tqdm


# Some quick helper func to make things easier

def to_1d_array(arr):
    return np.atleast_1d(arr)


def to_array(arr):
    return np.array(arr)


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


# some custom metrics on early enrichment
# (from https://chemrxiv.org/engage/chemrxiv/article-details/6585ddc19138d23161476eb1)

def plate_ppv(y, y_pred, top_n: int = 128):
    y_pred = np.atleast_1d(y_pred)
    y = np.atleast_1d(y)
    _tmp = np.vstack((y, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    _tmp = _tmp[np.where(_tmp[:, 1] > 0.5)[0]].copy()
    return np.sum(_tmp[:, 0]) / len(_tmp)


def diverse_plate_ppv(y, y_pred, clusters: list, top_n_per_group: int = 15):
    df = pd.DataFrame({"pred": y_pred, "real": y, "CLUSTER_ID": clusters})
    df_groups = df.groupby("CLUSTER_ID")

    _vals = []
    for group, idx in df_groups.groups.items():
        _tmp = df.iloc[idx].copy()
        if sum(df.iloc[idx]["pred"] > 0.5) == 0:
            continue
        _tmp = _tmp[_tmp["pred"] > 0.5].copy()
        _tmp = np.vstack((_tmp["real"].to_numpy(), _tmp["pred"].to_numpy())).T[_tmp["pred"].to_numpy().argsort()[::-1]][:top_n_per_group, :]
        _val = np.sum(_tmp[:, 0]) / len(_tmp)
        _vals.append(_val)

    return np.mean(_vals)


# some functions for generating the chemical fingerprints
class Basefpfunc(abc.ABC):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._func = None

    def __call__(self, chemicals, *args, **kwargs):
        return to_array([list(self._func(MolFromSmiles(c))) for c in to_1d_array(chemicals)])

    def __eq__(self, other):
        if isinstance(other, Basefpfunc):
            if inspect.signature(self._func).parameters == inspect.signature(other).parameters:
                return True
        return False

    def to_dict(self):
        _signature = inspect.signature(self._func)
        args = {
            k: v.default
            for k, v in _signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        args['name'] = self._func.func.__name__
        return args

    @property
    def __name__(self):
        return self._func.func.__name__


class ECFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self._func = partial(AllChem.GetHashedMorganFingerprint, **self._kwargs)


class ECFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self._func = partial(AllChem.GetHashedMorganFingerprint, **self._kwargs)


class FCFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self._func = partial(AllChem.GetHashedMorganFingerprint, **self._kwargs)


class FCFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self._func = partial(AllChem.GetHashedMorganFingerprint, **self._kwargs)


class BinaryECFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self._func = partial(AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryECFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self._func = partial(AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryFCFP4(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self._func = partial(AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class BinaryFCFP6(Basefpfunc):
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self._func = partial(AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)


class MACCS(Basefpfunc):
    def __init__(self):
        super().__init__()
        self._func = partial(rdMolDescriptors.GetMACCSKeysFingerprint, **self._kwargs)


class RDK(Basefpfunc):
    def __init__(self):
        super().__init__(**{"fpSize": 2048})
        self._func = partial(RDKFingerprint, **self._kwargs)


class Avalon(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(pyAvalonTools.GetAvalonCountFP, **self._kwargs)


class AtomPair(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(rdMolDescriptors.GetHashedAtomPairFingerprint, **self._kwargs)


class TopTor(Basefpfunc):
    def __init__(self):
        super().__init__(**{"nBits": 2048})
        self._func = partial(AllChem.GetHashedTopologicalTorsionFingerprint, **self._kwargs)


FPS_FUNCS = {'2048-ECFP4': ECFP4(),
             '2048-ECFP6': ECFP6(),
             '2048-FCFP4': FCFP4(),
             '2048-FCFP6': FCFP6(),
             '2048-bECFP4': BinaryECFP4(),
             '2048-bECFP6': BinaryECFP6(),
             '2048-bFCFP4': BinaryFCFP4(),
             '2048-bFCFP6': BinaryFCFP6(),
             'MACCS FP': MACCS(),
             'RDK FP': RDK(),
             'Avalon FP': Avalon(),
             'Atom Pair FP': AtomPair(),
             'Topological Torsion FP': TopTor()}


# this is some very unoptimized code, but the pickled model is built around it so I left it as is
# even if it lacks some efficiency and readability
class Model:
    def __init__(self):
        self._models = [[]]
        self._train_preds = []
        self._bayes = None
        self._fit = False
        self._ensemble = 0
        self._fp_func = []

    def fit(
            self,
            train_data: Dict[str, Union[npt.NDArray, str]],
            binary_labels: Union[npt.NDArray, str],
            clusters: Optional[Union[npt.NDArray, str]] = None,
            ensemble: int = 1
    ):
        """
        Fit the model
        :param train_data:
            should be a a dictionary where the key is the fingerprint type
            (see the `FPS_FUNCS` dict for names) and the val the path to a pickle
            or the loaded numpy array of the fingerprints

            Will make a separate model for each fingerprint type and mate. So if you set ensemble to 5
            use 4 different FPs, you will have 5 x 4 = 20 models
        :param binary_labels:
            the path to a pickle or the loaded numpy array of the binary labels
        :param clusters:
            the path to a pickle or the loaded numpy array of the cluster IDs
            not used if ensemble is <= 1
        :param ensemble:
            number of ensembles mates to use. Default is 1 (no ensemble)
        :return:
        """
        # load in pickles in needed
        for key, val in train_data.items():
            if isinstance(val, str):
                train_data[key] = pickle.load(open(val, "rb"))

        if isinstance(binary_labels, str):
            y = pickle.load(open(binary_labels, "rb"))
        else:
            y = binary_labels

        # save the fingerprints used for later
        self._fp_func = list(train_data.keys())

        if ensemble > 1:
            mates = []

            # load in cluster data
            if isinstance(clusters, str):
                clusters = pickle.load(open(clusters, "rb"))

            s = StratifiedGroupKFold(n_splits=ensemble, shuffle=True)
            for i, (train_idx, test_idx) in tqdm(enumerate(s.split(train_data[self._fp_func[0]], y, clusters)), desc="Doing Folds"):
                y_train = y[train_idx]
                models = []
                for _, x_train in train_data.items():
                    clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                    clf.fit(x_train, y_train)
                    models.append(deepcopy(clf))
                mates.append(models)
            self._models = deepcopy(mates)
        else:
            for _, x_train in train_data.items():
                clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                clf.fit(x_train, y)
                self._models[0].append(deepcopy(clf))

        self._fit = True
        self._ensemble = ensemble

    def screen(self, filepath, outpath: Optional[str] = None):
        """
        Screening a file of SMILES and returns predictions
        pred value will the be probability the model thinks something is active
        conf value is the confidence the model has in its predicted probability

        assumes you are using a smi file, so its tab delimited and first column is SMILES second is Name/ID

        :param filepath: path the .smi file to screen
        :param outpath: name of output file
        """

        if outpath is None:
            outpath = os.path.abspath(filepath).split('.')[0] + ".PREDS"

        with open(outpath, "w") as outfile:
            outfile.write("ID\tSMILES\tPRED\tCONF\n")

        with open(filepath, "r") as f:
            names = []
            smiles = []

            for i, line in tqdm(enumerate(f)):
                splits = line.split("\t")
                smiles.append(splits[0].strip())
                if len(splits) > 1:
                    names.append(splits[1].strip())
                else:
                    names.append(i)

                if ((i+1) % 100) == 0:
                    preds, confs = self.screen_smiles(smiles)
                    with open(outpath, "a") as f2:
                        for n, s, p, c in zip(names, smiles, preds, confs):
                            f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t{round(float(c), 4)}\n")
                        names = []
                        smiles = []

            # catch the last batch
            if len(smiles) != 0:
                preds, confs = self.screen_smiles(smiles)
                with open(outpath, "a") as f2:
                    for n, s, p, c in zip(names, smiles, preds, confs):
                        f2.write(f"{n}\t{s}\t{round(float(p), 4)}\t{round(float(c), 4)}\n")

    def screen_smiles(self, smis: list[str]):
        """
        Screens a list of smiles and returns predictions and confidences
        :param smis:
        :return:
        """
        fps = []
        for _fp in self._fp_func:
            fps.append(list(FPS_FUNCS[_fp](smis)))
        test_preds = []
        for i_model in range(self._ensemble):
            for clf, fp in zip(self._models[i_model], fps):
                test_preds.append(clf.predict_proba(fp)[:, 1])
        test_preds = np.array(test_preds).T
        preds = test_preds.mean(axis=1)
        confs = test_preds.std(axis=1)
        return preds, confs

    def cv(
            self,
            train_data: Dict[str, Union[npt.NDArray, str]],
            binary_labels: Union[npt.NDArray, str],
            clusters: Union[npt.NDArray, str],
            ensemble: int = 1,
    ):
        """
        Fit the model
        :param train_data:
            should be a a dictionary where the key is the fingerprint type
            (see the `FPS_FUNCS` dict for names) and the val the path to a pickle
            or the loaded numpy array of the fingerprints

            Will make a separate model for each fingerprint type and mate. So if you set ensemble to 5
            use 4 different FPs, you will have 5 x 4 = 20 models
        :param binary_labels:
            the path to a pickle or the loaded numpy array of the binary labels
        :param clusters:
            the path to a pickle or the loaded numpy array of the cluster IDs
            not used if ensemble is <= 1
        :param ensemble:
            number of ensembles mates to use. Default is 1 (no ensemble)
        :return:
        """
        # load in pickles in needed
        for key, val in train_data.items():
            if isinstance(val, str):
                train_data[key] = pickle.load(open(val, "rb"))

        if isinstance(binary_labels, str):
            y = np.array(pickle.load(open(binary_labels, "rb")))
        else:
            y = np.array(binary_labels)

        # load in cluster data
        if isinstance(clusters, str):
            clusters = pickle.load(open(clusters, "rb"))

        overall_res_ensemble = {
            "fit_time": [],
            "pred_time": [],
            "precision": [],
            "recall": [],
            "balanced_accuracy": [],
            "AUC_PR": [],
            "AUC_ROC": [],
            "PlatePPV": [],
            "DivPlatePPV": []
        }

        s = StratifiedShuffleSplit(test_size=0.2)

        for i, (train_idx, test_idx) in tqdm(enumerate(s.split(list(train_data.values())[0], y, clusters)), desc="Doing Folds"):
            y_train = y[train_idx]
            y_test = y[test_idx]

            train_clusters = clusters[train_idx]

            mates = []
            all_train_preds = []

            t0 = time()
            for _, x_train_ in train_data.items():
                x_train = x_train_[train_idx]
                if ensemble > 1:
                    # this is the ensemble builder
                    # should have done this so I could have reused the fit func but too late lol
                    s2 = StratifiedGroupKFold(n_splits=ensemble, shuffle=True)
                    models = []
                    train_preds = []

                    for ii, (train_idx2, test_idx2) in tqdm(enumerate(s2.split(x_train, y_train, train_clusters)), desc="Doing ensemble"):
                        clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                        x_train2 = x_train[train_idx2]
                        y_train2 = y_train[train_idx2]
                        clf.fit(x_train2, y_train2)
                        models.append(deepcopy(clf))
                        train_preds.append(clf.predict_proba(x_train)[:, 1])
                    mates.append(models)
                    all_train_preds.append(train_preds)

                else:
                    clf = LGBMClassifier(n_estimators=150, n_jobs=-1)
                    clf.fit(x_train, y_train)
                    mates.append([deepcopy(clf)])
                    all_train_preds.append([clf.predict_proba(x_train)[:, 1]])
            fit_time = time() - t0

            t0 = time()
            test_preds = []
            for clf_group, (_, x_test) in zip(mates, train_data.items()):
                x_test = x_test[test_idx]
                for clf in clf_group:
                    clf.predict_proba(x_test)
                    test_preds.append(clf.predict_proba(x_test)[:, 1])
            test_preds = np.array(test_preds).T
            pred_time = time() - t0

            preds = test_preds.mean(axis=1)
            discrete_preds = (preds > 0.3).astype(int)

            ppv = precision_score(y_test, discrete_preds)
            recall = recall_score(y_test, discrete_preds)
            auc_roc = roc_auc_score(y_test, preds)
            ba = balanced_accuracy_score(y_test, discrete_preds)
            auc_pr = average_precision_score(y_test, preds)
            p_ppv = plate_ppv(y_test, preds, top_n=128)
            dp_ppv = diverse_plate_ppv(y_test, preds, clusters=clusters[test_idx].tolist())

            overall_res_ensemble["fit_time"].append(fit_time)
            overall_res_ensemble["pred_time"].append(pred_time)
            overall_res_ensemble["precision"].append(ppv)
            overall_res_ensemble["recall"].append(recall)
            overall_res_ensemble["balanced_accuracy"].append(ba)
            overall_res_ensemble["AUC_ROC"].append(auc_roc)
            overall_res_ensemble["AUC_PR"].append(auc_pr)
            overall_res_ensemble["PlatePPV"].append(p_ppv)
            overall_res_ensemble["DivPlatePPV"].append(dp_ppv)

            print("ensemble", overall_res_ensemble)
        return pd.DataFrame(overall_res_ensemble)
