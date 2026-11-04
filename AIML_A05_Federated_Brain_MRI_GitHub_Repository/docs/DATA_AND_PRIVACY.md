# Data, privacy and research limitations

- The three Kaggle image-folder datasets are modelled as three simulated hospital/client cohorts. They are not evidence of real hospital participation.
- Dataset licences, terms, access dates and class-folder structures must be reviewed and recorded by the project team.
- The notebook uses perceptual hashing for image-level near-duplicate detection. This is not patient-level deduplication.
- Patient IDs are not guaranteed in the Kaggle cohorts. Do not claim patient-disjoint splits unless reliable IDs are present and used.
- The global test split is image-level unless the notebook has trustworthy case identifiers for a particular dataset.
- BraTS is an optional extension; its external evaluation is glioma-only and must not be presented as a four-class external validation.
- Differential privacy results depend on the exact clipping, sampling, adjacency and accountant assumptions. Report epsilon, delta and assumptions.
- Membership-inference and faulty-client tests are empirical experiments, not proof of security.
- Flower SecAgg+ scaffold generation is not proof that secure aggregation has been executed. Configure the maintained workflow/mod and complete the notebook's acceptance tests.
- Docker files and deployment scaffolds require actual builds and runtime tests.
- Brain MRI classification output is not a clinical diagnosis. This project is for research/education only.
