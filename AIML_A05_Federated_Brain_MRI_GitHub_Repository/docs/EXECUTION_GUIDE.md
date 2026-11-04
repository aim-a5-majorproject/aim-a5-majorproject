# Execution guide

## Colab-first execution

The notebook contains Colab-specific upload flows and `%pip` installation cells. Run it in Google Colab, preferably with a GPU runtime.

### Stage A — smoke test
- Run M0 setup and dependency installation.
- Keep `FULL_RESEARCH_RUN=False`.
- Authenticate to Kaggle using your own API token.
- Check all three source downloads and inspect the directory trees.
- Review class aliases and class counts before training.
- Inspect pHash de-duplication and the generated data card.
- Validate per-client train/validation/test splits and DataLoaders.
- Run short centralised/local/federated tests.

### Stage B — full experiments
- Set longer-run flags only after Stage A succeeds.
- Run the selected IID, natural-source and Dirichlet scenarios.
- Run the M5 optimizer/heterogeneity comparisons.
- Run M6 privacy, attack and robustness experiments with documented assumptions.
- Run M7 compression and communication experiments.
- Run M8 metrics, Grad-CAM, repeated seeds and paired statistics.
- BraTS cells are optional and require an authorised download.

### Stage C — deployment and audit
- Run M9 bundle generation.
- Build and test the API/dashboard containers.
- Treat the Flower multi-container project as a separate distributed deployment path.
- For SecAgg+, follow the notebook's acceptance test and retain server/client logs.
- Run final evidence audits only after the relevant artifacts and tests exist.

## Runtime notes
- Large CNNs, multiple scenarios and repeated-seed matrices can take substantial GPU time.
- Dataset download and access depend on Kaggle account permissions and current dataset terms.
- Save outputs before the Colab runtime is recycled.
- Never commit credentials, datasets or private user data.
