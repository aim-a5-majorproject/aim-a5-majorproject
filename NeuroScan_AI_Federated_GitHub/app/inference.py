from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from src.gradcam import make_gradcam, overlay_heatmap

IMG_SIZE = (224, 224)

class NeuroScanPredictor:
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train the Colab notebook and copy "
                "neuroscan_fedavg_vgg16.keras into the models/ directory."
            )
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_path, gradcam_output_path=None):
        image = load_img(image_path, target_size=IMG_SIZE)
        arr = img_to_array(image)
        original = arr.copy().astype(np.uint8)
        batch = preprocess_input(np.expand_dims(arr, 0))
        probability = float(self.model.predict(batch, verbose=0)[0][0])
        label = "Tumor Detected" if probability >= 0.5 else "No Tumor Detected"
        confidence = probability if probability >= 0.5 else 1.0 - probability

        gradcam_path = None
        if gradcam_output_path:
            heatmap = make_gradcam(batch, self.model)
            overlay = overlay_heatmap(original, heatmap)
            gradcam_path = Path(gradcam_output_path)
            gradcam_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gradcam_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "label": label,
            "tumor_probability": probability,
            "confidence": confidence,
            "gradcam_path": str(gradcam_path) if gradcam_path else None,
        }
