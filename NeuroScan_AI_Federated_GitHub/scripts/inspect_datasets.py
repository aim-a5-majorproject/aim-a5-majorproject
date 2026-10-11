import kagglehub
from src.data_utils import get_all_images, detect_binary_label

DATASETS = {
    "hospital_1": "masoudnickparvar/brain-tumor-mri-dataset",
    "hospital_2": "ahmedhamada0/brain-tumor-detection",
    "hospital_3": "navoneel/brain-mri-images-for-brain-tumor-detection",
}

for hospital, dataset in DATASETS.items():
    root = kagglehub.dataset_download(dataset)
    counts = {"tumor": 0, "no_tumor": 0, "unknown": 0}
    for image in get_all_images(root):
        label = detect_binary_label(image)
        counts[label if label else "unknown"] += 1
    print(hospital, dataset, root, counts)
