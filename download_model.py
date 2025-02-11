import gdown

file_ids = {
    "random_forest_model_download.pkl": "1giZkhUDPfzMSo1sHPKtk2-dRpBmrG9pA",
    "label_encoder_download.pkl": "1kGdnDlPQjNp8i5XrFbM-b13YypXYtAVn"
}

# Download each file
for filename, file_id in file_ids.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, filename, quiet=False)

print("All files downloaded successfully!")