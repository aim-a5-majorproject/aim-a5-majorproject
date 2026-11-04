# AIML-A05 — Privacy-Preserving Federated Learning for Brain Tumor MRI Classification

Research/education repository generated from the supplied 122-cell Google Colab notebook.

> **Canonical implementation:** [`notebooks/AIML_A05_COMPLETE_END_TO_END_COLAB.ipynb`](notebooks/AIML_A05_COMPLETE_END_TO_END_COLAB.ipynb)  
> This notebook is the source of truth. It contains the full M0–M9 implementation, optional extensions, deployment generators, result consolidation and evidence audits.

## Project overview

This project builds a research framework for four-class brain MRI image classification using three independently sourced Kaggle image cohorts as **simulated hospital clients**. It compares pooled centralised and independent local-only baselines with federated training, then studies non-IID behaviour, privacy, robustness, communication cost, explainability and deployment.

**Classes:** glioma, meningioma, pituitary, no_tumor.

**Important scope:** the three Kaggle cohorts are simulated clients, not verified hospitals. The image-folder datasets do not guarantee patient identifiers. Perceptual-hash duplicate filtering is image-level and must not be described as patient-level de-duplication. This is not a clinical diagnostic device.

## Repository structure

```text
AIML_A05_Federated_Brain_MRI_GitHub_Repository/
├── notebooks/
│   └── AIML_A05_COMPLETE_END_TO_END_COLAB.ipynb  # full 122-cell notebook
├── docs/
│   ├── MODULE_MAP.md
│   ├── EXECUTION_GUIDE.md
│   ├── DATA_AND_PRIVACY.md
│   └── RESEARCH_REPORT_TEMPLATE.md
├── configs/
│   └── default.yaml
├── scripts/
│   ├── inspect_notebook.py
│   └── verify_repository.py
├── deployment/
│   ├── api_dashboard/README.md
│   └── flower_multicontainer/README.md
├── data/README.md
├── outputs/README.md
├── requirements-colab.txt
├── requirements-deployment.txt
├── .gitignore
├── .env.example
└── LICENSE
```

The notebook creates the concrete model checkpoints, metrics, FastAPI service, Streamlit dashboard, Docker assets, Flower deployment scaffold and experiment evidence under its configured runtime output directories. Those generated artifacts are intentionally not fabricated or checked in here.

## Module coverage

| Module | Scope |
|---|---|
| M0 | Foundations, runtime/GPU, reproducibility, experiment configuration, member-level fine-tuning evidence |
| M1 | Kaggle authentication and three datasets, folder inspection, label harmonisation, pHash duplicate filtering, local/global splits, IID/natural/Dirichlet partitions, data cards, optional BraTS NIfTI extraction and case-level splitting |
| M2 | Image transforms, 224×224 inputs, grayscale-to-three-channel conversion, normalisation, augmentation, class weights, optional histogram matching and loader checks |
| M3 | ResNet50, EfficientNet-B0 and optional VGG16; feature extraction/fine-tuning; training utilities; local-only and centralised baselines |
| M4 | PyTorch FedAvg; client scenarios; per-round and per-hospital metrics; logical client expansion; Flower simulation |
| M5 | FedProx, FedAdam, FedBN with client-specific BatchNorm state, FedYogi and optional personalisation |
| M6 | Client update clipping/noise, client-level DP experiments and accounting, membership inference, robust aggregation, faulty-client tests, Flower SecAgg+ scaffold and acceptance guidance |
| M7 | Model/payload accounting, INT8 and top-k compression, communication-vs-accuracy analysis, local-epoch/round trade-off |
| M8 | Classification metrics, confusion matrix, per-client fairness, ROC, Grad-CAM, optional BraTS glioma-only external evaluation, three-seed runs and paired statistics |
| M9 | Checkpoints and metadata, Streamlit, FastAPI, Docker Compose, Flower multi-container project, reproducibility manifest and ZIP packaging |
| Final audits | Results consolidation, report skeleton, evidence-based module/action-plan audit and submission checklist |

See [`docs/MODULE_MAP.md`](docs/MODULE_MAP.md) for a detailed map of the notebook's named sections and extensions.

## Run the project

### Recommended: Google Colab

1. Upload/open the notebook in Google Colab.
2. Select a GPU runtime when available.
3. Set up Kaggle API access using your own `kaggle.json` through the notebook's upload prompt. **Never commit this file or its contents.**
4. Run cells from top to bottom.
5. Keep `FULL_RESEARCH_RUN=False` for the first smoke test. Confirm downloads, label mapping, data loaders and a short training run.
6. Only after the smoke test works, enable the longer research experiments as described in the notebook and execute the selected experiment matrix.
7. Run the final evidence audit after the corresponding experiments and deployment checks have actually completed.
8. Download the generated artifacts from the Colab runtime.

See [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md).

### Local environment

The notebook is Colab-oriented and contains Colab upload/runtime helpers and notebook magics. The supported primary path is Colab. `requirements-colab.txt` documents the principal Python dependencies; it is not a promise that every Colab-specific cell runs unchanged on every local operating system.

## Data access and credentials

The notebook obtains three Kaggle datasets via KaggleHub. You must authenticate with your own Kaggle credentials and accept applicable dataset terms. Dataset availability, licence, structure and access permission may change. No datasets or credentials are included in this repository.

BraTS 2021 is an optional, separately gated extension. Use only an authorised distribution and follow its access and licence requirements.

## Research integrity and completion criteria

A code cell or generated scaffold does not prove an experiment or deployment passed. Before claiming completion, retain actual evidence for:
- dataset access, class counts, deduplication report, splits and data card;
- training histories and matched baseline/federated comparisons;
- per-client metrics, confusion matrices, ROC and Grad-CAM outputs;
- privacy accounting assumptions and measured attack/robustness outcomes;
- serialized communication payload measurements;
- authorised BraTS case-disjoint external evaluation, if used;
- actual Flower SecAgg+ configuration and protocol/dropout acceptance tests, if claimed;
- successful Docker build and API/dashboard/Flower container tests;
- repeated-seed outputs and paired tests;
- final evidence audit.

See [`docs/DATA_AND_PRIVACY.md`](docs/DATA_AND_PRIVACY.md).

## GitHub publishing

1. Extract this ZIP.
2. Create a new empty GitHub repository.
3. Upload/push the extracted folder contents.
4. Confirm `.gitignore` is active and that no `kaggle.json`, downloaded dataset, private key, checkpoint, or personally identifiable data is committed.
5. Add your team authorship, institutional details and chosen licence before public release.

## Disclaimer

For research and education only. Not validated for clinical use, diagnosis, triage or treatment decisions. Dataset and deployment claims must be supported by actual execution evidence.
