"""Model creation, training, continued training (retraining) and evaluation.

The model is a small neural network (multi-layer perceptron) on top of HOG features of the
cleaned ECG trace. Standardisation and PCA are fitted once on the original training set and
then kept fixed, so a retrained network still sees the same feature space and can continue
from its saved weights instead of starting from scratch.
"""
import copy

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src import config


class EcgClassifier:
    def __init__(self, class_names, n_components=256, hidden=(512, 128), alpha=1e-2,
                 learning_rate=1e-3, seed=config.SEED):
        self.class_names = list(class_names)
        self.n_components = n_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=learning_rate,
                                 batch_size=32, random_state=seed)
        self.history = []          # one dict per epoch: train_loss, val_loss, val_accuracy

    # ------------------------------------------------------------ helpers
    def _encode(self, labels):
        idx = {c: i for i, c in enumerate(self.class_names)}
        return np.array([idx[l] for l in labels])

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.mlp.predict_proba(self.transform(X))

    def predict(self, X):
        return np.array(self.class_names)[self.predict_proba(X).argmax(axis=1)]

    # ------------------------------------------------------------ training
    def _train_epochs(self, Z, y, max_epochs, patience, val_fraction, log=None):
        """Epoch-by-epoch training with class weights, early stopping on validation loss,
        and restoring the best weights (like Keras' EarlyStopping(restore_best_weights=True))."""
        Ztr, Zval, ytr, yval = train_test_split(Z, y, test_size=val_fraction, stratify=y,
                                                random_state=self.seed)
        sw = compute_sample_weight("balanced", ytr)
        classes = np.arange(len(self.class_names))
        rng = np.random.RandomState(self.seed)
        best, best_loss, bad = None, np.inf, 0
        for epoch in range(max_epochs):
            order = rng.permutation(len(ytr))
            self.mlp.partial_fit(Ztr[order], ytr[order], classes=classes, sample_weight=sw[order])
            p_val = self.mlp.predict_proba(Zval)
            val_loss = log_loss(yval, p_val, labels=classes)
            rec = {"epoch": len(self.history) + 1, "train_loss": float(self.mlp.loss_),
                   "val_loss": float(val_loss), "val_accuracy": float(accuracy_score(yval, p_val.argmax(1)))}
            self.history.append(rec)
            if log:
                log(rec)
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best = (copy.deepcopy(self.mlp.coefs_), copy.deepcopy(self.mlp.intercepts_))
            else:
                bad += 1
                if bad >= patience:
                    break
        if best is not None:
            self.mlp.coefs_, self.mlp.intercepts_ = best
        return self

    def fit(self, X, labels, max_epochs=200, patience=15, val_fraction=0.15, log=None):
        """Train from scratch: fit the scaler and PCA, then the network."""
        y = self._encode(labels)
        Z = self.pca.fit_transform(self.scaler.fit_transform(X))
        return self._train_epochs(Z, y, max_epochs, patience, val_fraction, log)

    def continue_training(self, X, labels, learning_rate=3e-4, max_epochs=40, patience=8,
                          val_fraction=0.15, log=None):
        """Retraining: keep the saved scaler, PCA and network weights and keep training
        on the (old + new) data at a lower learning rate."""
        y = self._encode(labels)
        Z = self.transform(X)
        self.mlp.learning_rate_init = learning_rate
        if hasattr(self.mlp, "_optimizer"):
            del self.mlp._optimizer          # fresh Adam state at the new, lower learning rate
        return self._train_epochs(Z, y, max_epochs, patience, val_fraction, log)

    # ------------------------------------------------------------ saving
    def save(self, path=None):
        joblib.dump(self, path or config.MODEL_PATH)


def load_model(path=None):
    return joblib.load(path or config.MODEL_PATH)


def evaluate_model(model, X, labels):
    y_true = model._encode(labels)
    y_prob = model.predict_proba(X)
    y_pred = y_prob.argmax(axis=1)
    names = model.class_names
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    report = classification_report(y_true, y_pred, labels=range(len(names)), target_names=names,
                                   output_dict=True, zero_division=0)
    return {
        "test_loss": float(log_loss(y_true, y_prob, labels=range(len(names)))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "precision_weighted": float(wp),
        "recall_weighted": float(wr),
        "f1_weighted": float(wf1),
        "roc_auc_ovr_macro": auc,
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(names))).tolist(),
        "n_test": int(len(y_true)),
    }
