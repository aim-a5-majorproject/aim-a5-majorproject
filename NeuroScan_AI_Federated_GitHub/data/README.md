# Data

Raw datasets are not included in Git.

The supplied Colab notebook downloads:

- `masoudnickparvar/brain-tumor-mri-dataset`
- `ahmedhamada0/brain-tumor-detection`
- `navoneel/brain-mri-images-for-brain-tumor-detection`

It standardizes recognized folder labels to:

```text
hospital_1/
  no_tumor/
  tumor/
hospital_2/
  no_tumor/
  tumor/
hospital_3/
  no_tumor/
  tumor/
```

It then removes exact SHA-256 duplicates across hospitals and creates an independent 80/10/10 train/validation/test split for each hospital.

Do not upload patient-identifiable or restricted medical data to a public repository. Review each source dataset's license and terms independently.
