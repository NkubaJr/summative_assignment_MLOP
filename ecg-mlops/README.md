# ECG Image Classification: Machine Learning Cycle (MLOps)

An end-to-end machine learning pipeline that classifies scanned 12-lead ECG images into four heart conditions. It includes data processing, model training and evaluation, a FastAPI backend, a Streamlit UI, bulk upload with a retraining trigger, Docker deployment, cloud hosting and a Locust load test.

This extends my Introduction to ML summative (cardiovascular disease prediction from tabular data) to image data.

> Study project only. This is not a diagnostic tool.

## Links

-- **Video demo (YouTube):** https://youtu.be/Mo59oQXgkDo
- **Docker Hub image:** https://hub.docker.com/r/nkubajr/ecg-mlops

## Project description

**Task:** classify an ECG image as `Normal`, `Abnormal_Heartbeat`, `MI` (myocardial infarction) or `History_of_MI`.

**Data:** ECG Images dataset of Cardiac Patients (Khan and Hussain, Mendeley Data, v2), https://data.mendeley.com/datasets/gwbz3fsgp8/2. 928 images (one per patient), split 80/20 into `data/train` (742) and `data/test` (186), stratified by class.

**How the model works**

1. **Trace extraction.** Each sheet (2213 x 1572) is cropped to the ECG grid, the black trace is separated from the pink grid, leftover grid dots are removed, and the trace is thickened and resized to 384 x 256.
2. **HOG features.** The trace is described by the directions of its edges in 16 x 16 cells (12,420 numbers per image).
3. **Model.** Standardisation, PCA to 256 components, then a neural network (multi-layer perceptron, 512 and 128 hidden units) trained with class weights, L2 regularisation, the Adam optimiser and early stopping with best-weight restoring. Saved as `models/ecg_hog_mlp.pkl`.

My first attempt used MobileNetV2 transfer learning on whole sheets shrunk to 224 x 224 and reached only 33% test accuracy, because the thin trace disappears at that size. The notebook documents that attempt and why the pipeline changed.

## Model results (held-out test set, 186 images)

| Metric | MobileNetV2 (first attempt) | Final model (trace + HOG + MLP) |
|---|---|---|
| Accuracy | 0.328 | **0.946** |
| Precision (macro) | 0.157 | **0.947** |
| Recall (macro) | 0.323 | **0.940** |
| F1 (macro) | 0.206 | **0.943** |
| ROC-AUC (macro, one-vs-rest) | 0.685 | **0.993** |
| Log loss | 1.416 | **0.266** |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Abnormal_Heartbeat | 0.977 | 0.894 | 0.933 | 47 |
| History_of_MI | 0.909 | 0.882 | 0.896 | 34 |
| MI | 1.000 | 1.000 | 1.000 | 48 |
| Normal | 0.903 | 0.982 | 0.941 | 57 |

Every acute MI in the test set was caught, and nothing was wrongly labelled MI. The most common mistake was an old heart attack (`History_of_MI`) read as `Normal` (4 cases), which makes clinical sense: once an infarct has healed, the signs left on the ECG can be subtle.

In 5-fold cross-validation on the training set, an RBF SVM scored slightly higher (0.960 vs 0.931 accuracy). I deployed the neural network because it can continue training from its saved weights, which the retraining step needs.

## What the data shows

The notebook and the UI's Visualizations page interpret these features:

1. **Class balance.** Normal 284, MI 239, Abnormal_Heartbeat 233, History_of_MI 172. The 1.65 to 1 imbalance is mild, but it's why training uses class weights and the model is judged on macro F1.
2. **Trace ink density.** Abnormal-heartbeat sheets carry clearly more trace (median 3.34% of the grid vs about 2.6 to 2.8% for the others). More beats, extra beats and wider complexes all add ink. The other three classes overlap.
3. **Beat height in the rhythm strip.** 27% of abnormal-heartbeat sheets have deflections above 150 px, against 3% for MI and Normal. MI and Normal look alike here, because a heart attack mainly changes the ST segment and T wave in specific leads.
4. **Average sheet per class.** Normal traces stack tightly, MI is smeared across many leads, and History of MI shows a shadow below the baseline in leads V1 to V3.

The classes differ in *where and how the trace bends*, not in brightness or size (every image is 2213 x 1572). That's why position-aware edge features (HOG) work and shrinking the whole page did not.

## Repository structure

```
ecg-mlops/
├── README.md
├── notebook/
│   └── ecg_cardiac_mlops.ipynb   preprocessing, visualisations, training, evaluation, prediction, retraining check
├── src/
│   ├── preprocessing.py          trace extraction, HOG features, upload handling
│   ├── model.py                  EcgClassifier: train, continue training, evaluate, save/load
│   ├── prediction.py             predict_image() and the model store used by the API
│   ├── retrain.py                retraining pipeline and trigger
│   ├── database.py               SQLite records of uploads and retraining runs
│   └── config.py                 paths and settings
├── app/
│   ├── api.py                    FastAPI backend (Swagger docs at /docs)
│   └── ui.py                     Streamlit UI
├── data/
│   ├── train/                    742 images in one folder per class (see data/README.md)
│   └── test/                     186 images in one folder per class
├── models/
│   ├── ecg_hog_mlp.pkl           the trained model
│   ├── class_names.json
│   └── test_metrics.json
├── results/                      Locust CSV output
├── Dockerfile, docker-compose.yml, nginx.conf    local multi-container setup
├── Dockerfile.cloud, start.sh, render.yaml       single-container cloud deployment
├── locustfile.py, summarize_results.py           load testing
└── requirements.txt, requirements-locust.txt
```

## Setup

The commands below are for Git Bash on Windows; they also work on Mac and Linux terminals.

### Option 1: Docker (recommended)

1. Install Docker Desktop and start it.
2. Clone the repo and start everything:
   ```bash
   git clone <your repo URL>
   cd ecg-mlops
   docker compose up -d --build
   ```
3. Open the UI at http://localhost:8501 and the API docs at http://localhost:8000/docs.

### Option 2: Without Docker

Needs Python 3.11 or newer (3.13 matches the saved model's build).

```bash
python -m venv .venv
source .venv/Scripts/activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --port 8000      # terminal 1
streamlit run app/ui.py              # terminal 2
```

### Re-running the notebook

```bash
pip install jupyter
jupyter notebook notebook/ecg_cardiac_mlops.ipynb
```

It uses the images in `data/` and writes the model to `models/`. In Google Colab it can also download and split the raw dataset (section 2).

## Using the app

| UI page | What it does |
|---|---|
| Dashboard | API status, up-time, which container answered, test metrics and confusion matrix |
| Predict | Upload one ECG image and get the class, confidence and all four probabilities |
| Visualizations | The dataset features above, with their interpretations, rebuilt from the current data |
| Upload and Retrain | Bulk-upload labelled images, trigger retraining, watch progress and see the run history |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, up-time, container name, model last updated |
| `POST /predict` | one image in, class, confidence and probabilities out |
| `POST /upload` | several images plus their class label, saved to disk and recorded in the database |
| `POST /retrain?epochs=40` | start retraining |
| `GET /retrain/status` | progress of the current run |
| `GET /stats` | class counts, latest test metrics, retraining history |
| `GET /data/features` | per-image features used by the visualizations |

Example:
```bash
curl -F "file=@data/test/MI/MI(108).jpg" http://localhost:8000/predict
```

## How retraining works

1. New labelled images are uploaded in bulk (UI or `POST /upload`), saved under `data/uploads/<class>/` and recorded in SQLite.
2. **Trigger:** the "Trigger retraining" button, `POST /retrain`, or automatically once 20 new images are waiting (`RETRAIN_THRESHOLD`).
3. The new images are copied into `data/train/<class>/`.
4. Features are extracted with the same preprocessing as training (cached, so only new images are processed).
5. **The current saved model is loaded as the pretrained starting point.** Its scaler and PCA stay fixed and the network keeps training from its saved weights at a lower learning rate (0.0003), with early stopping.
6. The retrained model is evaluated on the fixed test set. If its macro F1 hasn't dropped by more than 0.02, the old model is archived in `models/archive/` and the new one goes live. All API containers pick it up automatically.

## Flood request simulation (Locust)

Each simulated user sends real test images to `POST /predict` (plus occasional `/health` checks) through nginx, which spreads requests over N API containers. 50 users, 60 seconds per run.

```bash
pip install -r requirements-locust.txt

docker compose up -d --scale api=1
locust -f locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --csv results/containers_1

docker compose up -d --scale api=2
locust -f locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --csv results/containers_2

docker compose up -d --scale api=3
locust -f locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --csv results/containers_3

python summarize_results.py
```

Results: can be found via the repo.

## Cloud deployment

`Dockerfile.cloud` packs the API, the UI, the model and the data into one container: the API runs inside on port 8000 and the UI is served on the public port.

**Render (free tier):** push the repo to GitHub, then on render.com choose New, then Blueprint, and select the repo. `render.yaml` sets up the service. The free tier sleeps after 15 minutes without traffic, so the first visit can take about a minute to wake it.

Live URL: can be found via the repo.

## Credits

Dataset: Khan, A. H. and Hussain, M. (2021), ECG Images dataset of Cardiac Patients, Mendeley Data, V2, doi:10.17632/gwbz3fsgp8.2
