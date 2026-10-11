# Federated Learning Design

The notebook uses three simulated clients and five federated rounds by default.

For every round:

1. Start with the current global weights.
2. Build one local model per hospital.
3. Load the global weights into the local model.
4. Train for the configured local epoch count.
5. Collect local weights and training sample counts.
6. Calculate a sample-weighted average of every trainable/non-trainable weight tensor.
7. Set the resulting weights on the global model.
8. Evaluate the global model on the concatenated validation datasets.

This is **FedAvg**, not FedProx. The supplied notebook does not implement a FedProx proximal term.

Because all clients execute inside the same Colab runtime, this is a research simulation of federated optimization rather than a distributed privacy infrastructure.
