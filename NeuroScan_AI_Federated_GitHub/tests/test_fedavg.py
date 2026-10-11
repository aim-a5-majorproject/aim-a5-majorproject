import numpy as np
from src.federated import fedavg

def test_fedavg_sample_weighting():
    client_a = [np.array([1.0, 3.0], dtype=np.float32)]
    client_b = [np.array([5.0, 7.0], dtype=np.float32)]
    result = fedavg([client_a, client_b], [1, 3])
    expected = np.array([4.0, 6.0], dtype=np.float32)
    assert np.allclose(result[0], expected)
