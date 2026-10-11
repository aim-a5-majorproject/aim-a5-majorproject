# Model Card — NeuroScan VGG16 FedAvg

## Intended use
Academic experimentation with federated-learning concepts for binary brain MRI image classification.

## Not intended for
Clinical diagnosis, treatment decisions, emergency triage, autonomous medical decisions, or claims of regulatory approval.

## Inputs
A 3-channel image resized to 224x224 and preprocessed with VGG16 preprocessing.

## Outputs
A sigmoid probability interpreted as Tumor when >= 0.5 and No Tumor otherwise.

## Training
Three simulated hospital clients; VGG16 frozen convolutional base; sample-weighted FedAvg.

## Explainability
Grad-CAM based on `block5_conv3`.

## Important limitations
Dataset composition, label quality, scanner/site variation, class imbalance, leakage, duplicates beyond exact byte matches, distribution shift, calibration, and external validation can affect results. Grad-CAM is not a tumor segmentation mask. Performance from one experiment must not be generalized to clinical populations without rigorous external validation.
