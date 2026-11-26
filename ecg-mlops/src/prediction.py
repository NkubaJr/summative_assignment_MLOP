import datetime
import os
import threading

import numpy as np

from src import config, preprocessing
from src.model import load_model


class ModelStore:
    """Holds the current model in memory and reloads it when the file on disk changes
    (after retraining, or when another container retrained it)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._mtime = None

    def _ensure_loaded(self):
        try:
            mtime = os.path.getmtime(config.MODEL_PATH)
        except OSError:
            raise RuntimeError("Model file not found. Run the notebook or train a model first.")
        if self._model is None or mtime != self._mtime:
            self._model = load_model(config.MODEL_PATH)
            self._mtime = mtime

    def model_version(self):
        try:
            ts = os.path.getmtime(config.MODEL_PATH)
        except OSError:
            return None
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")

    def predict_features(self, feats):
        with self._lock:
            self._ensure_loaded()
            model = self._model
        probs = model.predict_proba(feats[None, :])[0]
        idx = int(np.argmax(probs))
        return {
            "prediction": model.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(model.class_names, probs)},
        }

    def predict_bytes(self, data):
        return self.predict_features(preprocessing.features_from_bytes(data))

    def predict_path(self, path):
        return self.predict_features(preprocessing.features_from_rgb(preprocessing.load_rgb(path)))


store = ModelStore()


def predict_image(path):
    """Convenience function: predict one image file with the saved model."""
    return store.predict_path(path)


if __name__ == "__main__":
    import sys
    print(predict_image(sys.argv[1]))
