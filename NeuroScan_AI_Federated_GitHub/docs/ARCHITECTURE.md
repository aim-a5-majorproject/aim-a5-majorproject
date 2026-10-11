# System Architecture

## Training layer

The supplied notebook represents three Kaggle datasets as three simulated hospital clients. Each client receives the current global model weights and performs local VGG16 training. The local weights are returned to an in-process aggregator.

## Aggregation layer

The aggregator uses sample-weighted FedAvg:

`global_weight = sum(client_weight_i * client_sample_count_i / total_samples)`

The updated global weights are used in the next federated round.

## Model layer

VGG16 uses ImageNet initialization in the first global model. The convolutional base is frozen. The custom head is GlobalAveragePooling -> Dense 256 -> Dense 128 -> Sigmoid.

## Evaluation layer

The final global model is evaluated on the concatenation of the three hospitals' test datasets and independently on every hospital. Metrics include accuracy, precision, recall, F1 and ROC-AUC.

## Explainability layer

Grad-CAM is computed from VGG16 `block5_conv3` and overlaid on the uploaded MRI.

## Demo layer

The Flask demo loads the `.keras` global model and performs inference. It does not implement federated training or clinical workflow integration.

## Production boundary

The notebook simulates federation in one runtime. Production multi-hospital federation would require secure networking, authenticated clients, privacy/security controls, auditability, deployment governance, monitoring, and medical validation that are outside this repository.
