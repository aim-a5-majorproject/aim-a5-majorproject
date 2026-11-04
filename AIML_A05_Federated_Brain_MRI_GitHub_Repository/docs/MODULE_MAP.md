# Notebook module map

This map is based on the uploaded notebook's actual headings and named code cells. The notebook remains the canonical implementation.

- Notebook cells: **122**
- Non-empty code cells: **79**

## Notebook map — complete module plan
| Module | Included work | |---|---| | M0 | CNN/transfer-learning foundations, MRI class context, FL concepts, GPU and experiment tracking | | M1 | Three Kaggle datasets, integrity checks, perceptual duplicate detection, hospital scenarios, fixed global/local test sets, optional BraTS | | M2 | Brain-region crop option, 224×224, grayscale-to-RGB, per-image normalization, augmentation, class weights, per-client loaders | | M3 | Centralized/local-only baselines; ResNet50, EfficientNetB0, optional VGG16; feature extraction/fine-tuning | | M4 | FedAvg simulation, local epochs, client fraction, rounds, IID/natural/non-IID experiments, per-round tra

## M0 — Foundations, environment and experiment settings
**Review-II mathematics:** convolution, softmax cross-entropy and FedAvg intuition.  For a convolution, `Y[i,j] = sum_(u,v) X[i+u,j+v] K[u,v]`. For classification, use softmax probabilities and cross-entropy. FedAvg combines client weights using their sample counts: `w_global = Σ_k (n_k / n) w_k`.

## M1 — Dataset acquisition and three-hospital simulation
KaggleHub is used to download each dataset independently. Authenticate with a Kaggle API token. In Colab, create a Kaggle API token from your Kaggle account settings and upload `kaggle.json` when prompted. Do not paste or publish API keys in a notebook.

## M2 — Preprocessing and harmonisation
All clients use the same preprocessing pipeline: resize to 224×224, grayscale-to-three-channel conversion, per-image intensity normalisation and training-only augmentation. A simple center crop is provided as an optional brain-region crop; it is not a validated skull-stripping algorithm. Histogram matching is optional and disabled by default.

## M3 — Centralised and local-only baselines
Compare ResNet50, EfficientNetB0 and optional VGG16. Feature extraction freezes the pretrained backbone; fine-tuning unfreezes it. Hyperparameters must remain fixed when comparing centralised, local-only and federated runs.

## M4 — Federated training with FedAvg
This section implements a transparent PyTorch FedAvg simulation with three clients. Raw image data remains in each client DataLoader; only model parameters are returned to the coordinator. For a production-grade distributed deployment, use Flower's client/server runtime and transport/security configuration.

## M5 — Non-IID heterogeneity: FedProx, FedBN and server optimizers
FedProx adds `(μ/2)||w − w_global||²` to each client's local objective. FedBN keeps BatchNorm parameters/statistics local to handle feature-shift/scanner differences. FedAdam/FedYogi are server-side adaptive optimizers. The compact reference implementation below includes FedProx and a FedAdam-style server update; FedBN is provided as a local-BN aggregation policy template. Validate against a pinned Flower version before reporting as a production implementation.

## M6 — Privacy and security layer
Differential privacy requires clipping each client update and adding calibrated noise. Epsilon depends on the sampling scheme, number of rounds, clipping norm, noise multiplier and accountant. The simple Gaussian calculation below is an educational approximation only; use Opacus/Flower's supported accountant for a defensible privacy claim. Secure aggregation is a cryptographic protocol; averaging masked updates in a notebook is not a substitute.

## M7 — Communication efficiency
Measure transmitted model-state bytes, round count and estimated communication. Compare ResNet50 and EfficientNetB0 parameter counts. 8-bit quantisation and top-k sparsification below are payload-compression demonstrations; evaluate accuracy impact after dequantisation.

## M8 — Evaluation, explainability, fairness and statistics

## M9 — Hospital dashboard, diagnosis demo and reproducible reporting
This module creates a Streamlit dashboard template, saves model/results artifacts and writes a Docker Compose example. The notebook's model is a research classifier; the demo must display a clear non-clinical disclaimer.

## Final report and paper checklist (aligned with AIML-A05 plan)
- [ ] State research question, hypotheses and contribution. - [ ] Cite the base paper: Albalawi et al., *Integrated Approach of Federated Learning with Transfer Learning for Classification and Diagnosis of Brain Tumor*, BMC Medical Imaging 24:110 (2024). - [ ] Discuss supporting work: Lee et al. (Nature Communications, 2024); Lhasnaoui et al. (KDD FL Workshop, 2024); Ay & Ekinci (IJIST, 2024); Gohari et al., FedBrain-Distill (2024). - [ ] Record exact Kaggle dataset versions, download dates, licenses, class counts and preprocessing. - [ ] Clearly disclose that clients are simulated cohorts and patient-level IDs may be unavailable. - [ ] Repor

## Action-plan coverage map — exact sub-module checklist
| Plan item | Implementation location / evidence | |---|---| | 0.1–0.4 | M0 sections, model fine-tuning and experiment metadata | | 1.1–1.5 | Kaggle cohorts, BraTS NIfTI import, patient/case grouping, deduplication, IID/natural/Dirichlet partitions, fixed tests | | 2.1–2.4 | crop/resize/RGB/z-score, histogram matching option, augmentation, class weights and loaders | | 3.1–3.5 | pooled baseline, per-client local baseline, ResNet50/EfficientNetB0/VGG16, feature-extract vs fine-tune | | 4.1–4.5 | Flower simulation with 4–10 logical clients, local epochs, partial participation, scenario and per-client tracking | | 5.1–5.5 | FedProx, true client-

## M0 completion — member-level fine-tuning evidence and MRI modality notes
The plan's M0 deliverable requires each team member to fine-tune a pretrained CNN on a small MRI subset and complete an FL short course. The notebook can record evidence, but each member must perform their own run and course. T1 is anatomical; T1ce/T1Gd is post-contrast; T2 highlights fluid-sensitive structures; FLAIR suppresses free-fluid signal. The Kaggle JPEG datasets generally do not retain reliable modality metadata, so do not label them T1/T2/FLAIR unless the source documentation confirms it.

## M1 completion — BraTS 2021 access, NIfTI indexing, case-level split and external glioma cohort
BraTS 2021 Task 1 contains multi-parametric NIfTI MRI and segmentation masks. The official challenge page explains registration and access; the Kaggle mirror may also require acceptance of dataset terms. Set the Kaggle slug only after confirming that it is the intended BraTS 2021 Task 1 dataset and that your account has access. A BraTS case is glioma-only; it is **not** a four-class external test for meningioma/pituitary/no-tumor. We therefore report it separately as a glioma-only external cohort and avoid pretending it supplies the other three labels.

## M2 completion — optional histogram harmonisation and reusable loader validation
The plan makes histogram matching optional. This cell demonstrates matching each image to a fixed reference histogram. Record the reference image and whether the transformation improved or harmed validation performance; do not apply it blindly. For real scanner harmonisation, scanner metadata and site-specific validation are preferable.

## M3 completion — explicit architecture and feature-extraction/fine-tuning comparison

## M4 completion — 4–10 Flower clients and partial participation
The original notebook's manual FedAvg simulation is retained for transparent algorithm inspection. The next cell provides a real Flower simulation runner using Flower's simulation API. To meet the plan's 4–10-client requirement while keeping three source cohorts, each source cohort is deterministically split into two logical clients (six total). This remains a *simulation*, not six independent hospitals. Pin the Flower version for reproducibility; Flower APIs can change between releases.

## M5 completion — FedYogi and true client-specific FedBN state
FedBN must retain each client's BatchNorm parameters/statistics locally; averaging all BN tensors or keeping only the first client's BN state is not a valid FedBN implementation. The following runner maintains a private BN state dictionary per simulated client and aggregates only non-BN parameters. FedYogi is provided as an adaptive server-optimizer option.

## M6 completion — formal client-level DP accounting, membership-inference classifier, and SecAgg+ deployment scaffold
**Privacy distinction:** the previous notebook's hand-written epsilon heuristic is not a formal DP guarantee. This section uses Opacus' RDP accountant for a client-subsampled Gaussian mechanism. It is a client-level model-update DP experiment, not patient-level DP. The privacy claim is valid only under the stated adjacency, clipping, sampling and accountant assumptions. Do not combine ε values from unrelated runs.

## M7 completion — measured communication accounting and accuracy/communication trade-off

## M8 completion — BraTS external glioma test, Grad-CAM, 3 independent seeds and paired statistics

## M9 completion — dashboard, Grad-CAM inference, API, Docker and reproducible project bundle
The previous Docker Compose file referenced missing `server.py` and `client.py`; this section replaces it with a runnable research-demo bundle for the trained model, dashboard and API. It is a single-host demonstration. For a genuine multi-site deployment, each hospital must run its client in its own controlled network and connect to a secured Flower server; do not expose patient data or credentials in this demo.

## M4–M8 result consolidation — save comparison tables and experiment logs

## Final module-action-plan audit — only mark PASS when evidence exists
This audit records whether the required artifacts exist. It deliberately does not mark a requirement complete merely because a code cell exists. Run after completing the experiments and deployment. Any `MISSING` item must be addressed before claiming the module deliverable is complete.

## Submission checklist and reporting rules
- Include the data card, exact dataset URLs/slugs, access date, license/terms, preprocessing, split logic, class counts and deduplication report. - State clearly that Kaggle cohorts are simulated hospitals and disclose possible source overlap. Perceptual hashing is not patient-level de-duplication. - Report each scenario, model, method, seed, client fraction, local epochs, rounds, hardware, runtime and package versions. - Include centralised, local-only and federated comparisons; FedAvg/FedProx/FedBN/FedAdam/FedYogi; DP ε/δ and assumptions; SecAgg+ protocol evidence; MIA AUC; robustness test; communication costs; per-hospital fairness; Grad-C

## Final end-to-end execution, evidence packaging and submission audit
Run cells from top to bottom in a fresh Colab GPU runtime. Start with `FULL_RESEARCH_RUN=False` to validate data access, preprocessing, and a short training cycle. After the smoke test succeeds, set it to `True` in M0.3 and rerun the notebook to enable the longer architecture, communication trade-off, and three-seed experiments. BraTS requires authorised access and is deliberately gated. Flower SecAgg+ requires a separately executed Flower application and cannot be honestly marked complete by generating a template. Docker Compose must be built and run on a Docker-capable host; Colab does not guarantee a Docker daemon.  **Research integrity:**

## Named implementation cells

- Cell 4: `# M0.1 — Runtime, GPU and reproducibility`
- Cell 5: `# M0.2 — Install required libraries`
- Cell 6: `# M0.3 — Experiment configuration (adjust after a smoke test)`
- Cell 9: `# M1.1 — Kaggle authentication`
- Cell 10: `# M1.2 — Download three datasets separately using KaggleHub`
- Cell 11: `# M1.3 — Inspect folder structures before making assumptions`
- Cell 13: `CLASS_ALIASES = {`
- Cell 14: `# M1.5 — Perceptual-hash duplicate detection across hospital cohorts`
- Cell 15: `# M1.6 — Hospital-local train/validation/test splits and fixed global test set`
- Cell 16: `# M1.7 — IID, natural-source and Dirichlet label-skew partition scenarios`
- Cell 18: `# M1.8 — Data card export`
- Cell 20: `# Optional BraTS NIfTI slice extraction (run only after authorized download)`
- Cell 22: `# M2.1 — Dataset and transforms`
- Cell 23: `# M2.2 — Class weights w_k = N / (K * n_k)`
- Cell 26: `# M3.1 — Model factory`
- Cell 27: `# M3.2 — Training and evaluation utilities`
- Cell 28: `# M3.3 — Local-only baseline: one independent model per hospital`
- Cell 29: `# M3.4 — Centralised upper-bound baseline (pooled training data; baseline only)`
- Cell 32: `# M4.1 — State-dict helpers and weighted FedAvg`
- Cell 33: `# M4.2 — Compare IID, natural-source and Dirichlet label-skew scenarios`
- Cell 34: `# M4.3 — Per-round global and per-hospital accuracy plot`
- Cell 36: `# M5.1 — FedProx experiment`
- Cell 37: `# M5.2 — FedAdam server optimizer (pseudo-gradient on model deltas)`
- Cell 38: `# M5.3 — FedBN aggregation policy helper`
- Cell 39: `# M5.4 — Optional personalized fine-tuning per hospital`
- Cell 42: `# M6.1 — Client update clipping and Gaussian noise (central coordinator simulation)`
- Cell 43: `# M6.2 — DP FedAvg demonstration`
- Cell 44: `# M6.3 — Membership inference demonstration (shadow-free confidence threshold baseline)`
- Cell 45: `# M6.4 — Robust aggregation: coordinate-wise median and trimmed mean`
- Cell 48: `# M7.1 — Model size and communication accounting`
- Cell 49: `# M7.2 — 8-bit symmetric quantisation and top-k sparsification utilities`
- Cell 51: `# M8.1 — Full metric report and confusion matrix`
- Cell 52: `# M8.2 — Per-hospital fairness and worst-hospital accuracy`
- Cell 53: `# M8.3 — ROC curves (one-vs-rest)`
- Cell 54: `# M8.4 — Grad-CAM gallery for ResNet50 (run if the installed package supports this model)`
- Cell 55: `# M8.5 — Repeated seeds and mean ± standard deviation (compact protocol)`
- Cell 56: `# M8.6 — Paired test template (requires per-seed paired metric arrays for both methods)`
- Cell 59: `# M9.1 — Save model, metrics, histories and experiment metadata`
- Cell 60: `# M9.2 — Generate a Streamlit inference/dashboard app`
- Cell 61: `# M9.3 — Docker Compose template for server + 3 hospital client services`
- Cell 62: `# M9.4 — Package notebook outputs for download`
- Cell 67: `# M0.5 — reproducible individual fine-tuning evidence (run separately for each team member)`
- Cell 69: `# M1.10 — Download BraTS 2021 from KaggleHub (requires accepted access + Kaggle credentials)`
- Cell 70: `# M1.11 — robust NIfTI case discovery and T1ce/FLAIR slice extraction with patient/case IDs`
- Cell 71: `# M1.12 — case/patient-level split utility (use case_id where it exists; never split slices from one case across folds)`
- Cell 72: `# M1.13 — save data cards and partition files for all scenarios`
- Cell 74: `# M2.5 — histogram matching transform (grayscale reference)`
- Cell 75: `# M2.6 — preprocessing audit and loader smoke test`
- Cell 77: `# M3.6 — architecture parameter/size table and trainable-vs-frozen experiment runner`
- Cell 78: `# M3.7 — controlled baseline runner: pooled, local-only and model variants`
- Cell 80: `# M4.6 — create 6 logical clients from the three hospital cohorts`
- Cell 81: `# M4.7 — install pinned Flower simulation dependency`
- Cell 82: `# M4.8 — executable Flower FedAvg simulation (run one scenario first)`
- Cell 84: `# M5.6 — correct FedBN state partition and client-specific BN persistence`
- Cell 85: `# M5.7 — FedYogi server optimiser (adaptive second moment)`
- Cell 87: `# M6.5 — install Opacus and compute formal RDP accountant epsilon for client-level DP`
- Cell 88: `# M6.6 — client-level DP FedAvg with clipping of each hospital update and noise on the aggregated sum`
- Cell 89: `# M6.7 — stronger empirical membership-inference attack using confidence, entropy and correctness features`
- Cell 90: `# M6.8 — faulty-client robustness experiment (synthetic update corruption)`
- Cell 91: `# M6.9 — generate official Flower SecAgg+ example project scaffold (actual secure protocol, not a mock)`
- Cell 94: `# M7.6 — per-round communication estimator from actual model tensor sizes and compression factors`
- Cell 95: `# M7.7 — communication vs accuracy plot from available experiment histories`
- Cell 96: `# M7.8 — apply compression to client model deltas and measure actual serialized payload bytes`
- Cell 97: `# M7.9 — local-epoch / communication-round trade-off experiment`
- Cell 99: `# M8.7 — BraTS external glioma-only evaluation (one-class cohort; report sensitivity/recall only)`
- Cell 100: `# M8.8 — Grad-CAM for ResNet50 / EfficientNetB0 / VGG16 using architecture-specific target layers`
- Cell 101: `# M8.9 — proper repeated-seed protocol (3 seeds × selected methods/scenario)`
- Cell 102: `# M8.10 — paired tests from matched seeds only`
- Cell 104: `# M9.5 — create a self-contained deployment bundle with model, history, app, API, Docker and compose`
- Cell 105: `# M9.6 — create FastAPI inference service (research demonstration; no clinical use)`
- Cell 106: `# M9.7 — create Streamlit dashboard with metrics and image upload; Grad-CAM gallery is linked as an output artifact`
- Cell 107: `# M9.8 — requirements and Docker Compose bundle`
- Cell 109: `# M9.10 — final reproducibility manifest and zipped submission bundle`
- Cell 111: `# M9.11 — write server.py/client.py and compose file for Flower multi-container demo`
- Cell 113: `# Save consolidated research tables after running the corresponding experiments`
- Cell 115: `# Rebuild the final ZIP after the Flower multi-container project has been generated`
- Cell 117: `# FINAL AUDIT — evidence-based status`
- Cell 120: `# FINAL-1 — generate report skeleton, dataset/evidence manifest and execution checklist`
- Cell 121: `# FINAL-2 — stricter action-plan evidence audit (file existence is necessary, not sufficient)`