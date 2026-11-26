# Data

The 928 ECG images (about 590 MB) are left out of this zip because of the upload size limit. They are in the GitHub repo.

To rebuild the folders yourself, download the dataset from https://data.mendeley.com/datasets/gwbz3fsgp8/2 and run section 2 of the notebook in Google Colab. It splits the images 80/20 (stratified, seed 42) into:

- `data/train/` – 742 images: Abnormal_Heartbeat 186, History_of_MI 138, MI 191, Normal 227
- `data/test/` – 186 images: Abnormal_Heartbeat 47, History_of_MI 34, MI 48, Normal 57

The trained model (`models/ecg_hog_mlp.pkl`) and the notebook outputs are already included, so predictions work without the images.
