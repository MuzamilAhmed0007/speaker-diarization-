import kagglehub

# Download latest version
path = kagglehub.dataset_download("washingtongold/voxconverse-dataset")

print("Path to dataset files:", path)
