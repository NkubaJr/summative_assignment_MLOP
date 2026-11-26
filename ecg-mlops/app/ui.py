import datetime
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="ECG Classifier", layout="wide")
sns.set_theme(style="whitegrid")
COLORS = {"Abnormal_Heartbeat": "#257de0", "History_of_MI": "#f47230", "MI": "#00b894", "Normal": "#f2a600"}


def api_get(path, timeout=60):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        return None


def pretty_seconds(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


st.sidebar.title("ECG Classifier")
page = st.sidebar.radio("Navigate", ["Dashboard", "Predict", "Visualizations", "Upload and Retrain"])
st.sidebar.caption("Classifies 12-lead ECG images into Normal, Abnormal Heartbeat, MI and History of MI. "
                   "A study project, not a diagnostic tool.")


# ------------------------------------------------------------------ Dashboard
if page == "Dashboard":
    st.title("Model dashboard")
    health = api_get("/health")
    stats = api_get("/stats")
    if health:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API status", health["status"].upper())
        c2.metric("Up-time", pretty_seconds(health["uptime_seconds"]))
        c3.metric("Model loaded", "Yes" if health["model_loaded"] else "No")
        c4.metric("Served by container", health["served_by"])
        st.caption(f"API started at {health['started_at']} (UTC). Model file last updated: {health['model_last_updated']}")

    if stats and stats.get("metrics"):
        m = stats["metrics"]
        st.subheader("Model performance on the held-out test set")
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        cols[1].metric("Precision (macro)", f"{m['precision_macro']:.3f}")
        cols[2].metric("Recall (macro)", f"{m['recall_macro']:.3f}")
        cols[3].metric("F1 (macro)", f"{m['f1_macro']:.3f}")
        auc = m.get("roc_auc_ovr_macro")
        cols[4].metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "n/a")
        st.caption(f"Source: {m.get('source', 'notebook')}, updated {m.get('updated_at', 'n/a')}, "
                   f"{m.get('n_test', '?')} test images.")
        left, right = st.columns(2)
        if m.get("per_class"):
            left.write("Per-class results")
            left.dataframe(pd.DataFrame(m["per_class"]).T.round(3))
        if m.get("confusion_matrix"):
            names = list(m["per_class"].keys())
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=names, yticklabels=names, linewidths=2, linecolor="white", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion matrix", loc="left")
            plt.xticks(rotation=20)
            right.pyplot(fig)
    if st.button("Refresh"):
        st.rerun()


# ------------------------------------------------------------------ Predict
elif page == "Predict":
    st.title("Predict from a single ECG image")
    uploaded = st.file_uploader("Choose an ECG image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        left, right = st.columns([1, 1])
        left.image(uploaded, caption=uploaded.name)
        if right.button("Predict", type="primary"):
            with st.spinner("Running the model..."):
                try:
                    r = requests.post(f"{API_URL}/predict",
                                      files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                                      timeout=120)
                    r.raise_for_status()
                    res = r.json()
                    right.success(f"Prediction: {res['prediction']}  (confidence {res['confidence']:.1%})")
                    right.caption(f"Latency {res['latency_ms']} ms, served by {res['served_by']}")
                    right.bar_chart(pd.Series(res["probabilities"], name="probability"))
                except Exception as exc:  # noqa: BLE001
                    right.error(f"Prediction failed: {exc}")


# ------------------------------------------------------------------ Visualizations
elif page == "Visualizations":
    st.title("Data visualizations")
    st.caption("Built live from the images currently in data/train and data/test, so they update after new data is added.")
    stats = api_get("/stats")
    with st.spinner("Reading the dataset (about 30 seconds the first time)..."):
        feats = api_get("/data/features", timeout=600)

    if stats and stats["train_counts"]:
        counts = pd.DataFrame({"train": stats["train_counts"], "test": stats["test_counts"]}).fillna(0)
        counts["total"] = counts["train"] + counts["test"]
        counts = counts.sort_values("total", ascending=False)

        st.subheader("1. Class distribution")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(counts.index, counts["total"], color=[COLORS.get(c, "#257de0") for c in counts.index], width=0.6)
        ax.bar_label(bars, padding=3)
        ax.set_ylabel("Images")
        ax.grid(axis="x", visible=False)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        top, bottom = counts["total"].idxmax(), counts["total"].idxmin()
        st.write(f"**Story:** {top} is the largest class ({int(counts['total'].max())} images) and {bottom} the "
                 f"smallest ({int(counts['total'].min())}), about {counts['total'].max() / counts['total'].min():.2f} to 1. "
                 f"The imbalance is mild, but a model could still lean towards the big class, so training uses class "
                 f"weights and the model is judged on macro F1 and per-class recall, not accuracy alone.")

    if feats:
        df = pd.DataFrame(feats)
        order = [c for c in COLORS if c in df["label"].unique()] + sorted(set(df["label"]) - set(COLORS))

        st.subheader("2. How much heart trace is on each sheet")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.boxplot(data=df, x="label", y="trace_ink_pct", order=order, palette=COLORS, width=0.55, fliersize=3, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Trace ink (% of grid area)")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        med = df.groupby("label")["trace_ink_pct"].median()
        others = med.drop(med.idxmax())
        st.write(f"**Story:** {med.idxmax()} sheets carry the most trace (median {med.max():.2f}% of the grid) "
                 f"against {others.min():.2f} to {others.max():.2f}% for the rest. Abnormal rhythms mean more beats, "
                 f"extra beats or wider complexes, and each puts more ink on the page. The other classes overlap, so "
                 f"ink alone can spot abnormal rhythms but can't separate MI, History of MI and Normal.")

        st.subheader("3. Beat height in the lead II rhythm strip")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.boxplot(data=df, x="label", y="rhythm_amplitude_px", order=order, palette=COLORS, width=0.55, fliersize=3, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Tallest deflection (px)")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        tall = (df["rhythm_amplitude_px"] > 150).groupby(df["label"]).mean()
        st.write(f"**Story:** {tall.idxmax()} has the tallest, most variable beats: {tall.max():.0%} of its sheets go "
                 f"above 150 px, against {tall.min():.0%} for {tall.idxmin()}. MI and Normal look almost the same here, "
                 f"because a heart attack mainly changes the ST segment and T wave in specific leads, not beat height. "
                 f"That's why the model uses HOG features, which keep track of where and how the trace bends.")

        st.subheader("4. Image size")
        sizes = df[["width", "height"]].drop_duplicates()
        st.write(f"**Story:** the dataset has {len(sizes)} distinct image size(s) "
                 f"({', '.join(f'{w} x {h}' for w, h in sizes.values[:3])}). Shrinking a page this big to 224 x 224 "
                 f"erases the thin trace, which is why the first CNN failed. The pipeline crops to the grid, keeps only "
                 f"the trace and resizes to 384 x 256 instead.")


# ------------------------------------------------------------------ Upload and Retrain
else:
    st.title("Upload new data and retrain")
    stats = api_get("/stats")
    status = api_get("/retrain/status")

    if stats:
        st.subheader("1. Upload labelled images")
        st.caption("Choose the class these images belong to, then select several image files at once.")
        label = st.selectbox("Class label", stats["classes"])
        files = st.file_uploader("ECG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button("Upload images", disabled=not files):
            payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            try:
                r = requests.post(f"{API_URL}/upload", data={"label": label}, files=payload, timeout=300)
                r.raise_for_status()
                res = r.json()
                st.success(f"Saved {res['saved']} image(s). Pending for retraining: {res['pending_uploads']} "
                           f"(auto-retrain starts at {res['retrain_threshold']}).")
                if res["rejected"]:
                    st.warning(f"Rejected (not valid images): {res['rejected']}")
                if res["auto_retrain_started"]:
                    st.info("The pending-upload threshold was reached, so retraining started automatically.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")

        st.subheader("2. Retrain")
        c1, c2 = st.columns(2)
        c1.metric("Pending new images", stats["pending_uploads"])
        c2.metric("Auto-retrain threshold", stats["retrain_threshold"])
        if stats["pending_by_label"]:
            st.write("Pending by class:", stats["pending_by_label"])
        epochs = st.slider("Maximum epochs (early stopping usually ends sooner)", 5, 100, 40)
        running = bool(status and status.get("running"))
        if st.button("Trigger retraining", type="primary", disabled=running):
            started = False
            try:
                r = requests.post(f"{API_URL}/retrain", params={"epochs": epochs}, timeout=60)
                r.raise_for_status()
                started = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not start retraining: {exc}")
            if started:
                time.sleep(1)
                st.rerun()

    if status:
        st.subheader("3. Retraining status")
        st.write(f"**Stage:** {status['stage']}")
        st.write(status["message"])
        if status["running"]:
            st.info("Retraining is running. This page refreshes every few seconds.")
            time.sleep(3)
            st.rerun()

    if stats and stats["retrain_runs"]:
        st.subheader("Retraining history")
        runs = pd.DataFrame(stats["retrain_runs"])[
            ["id", "started_at", "finished_at", "status", "n_new_images", "accuracy", "f1_macro", "notes"]]
        st.dataframe(runs, hide_index=True)
