from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from app.inference import NeuroScanPredictor

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "app" / "static" / "uploads"
MODEL_PATH = ROOT / "models" / "neuroscan_fedavg_vgg16.keras"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    predictor = None
    model_error = None
    try:
        predictor = NeuroScanPredictor(MODEL_PATH)
    except Exception as exc:
        model_error = str(exc)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = model_error
        image_url = None
        gradcam_url = None

        if request.method == "POST":
            if predictor is None:
                return render_template("index.html", error=error)

            file = request.files.get("mri")
            if not file or not file.filename:
                error = "Please choose an MRI image."
            elif not allowed_file(file.filename):
                error = "Only PNG, JPG, and JPEG images are accepted."
            else:
                suffix = Path(secure_filename(file.filename)).suffix.lower()
                token = uuid4().hex
                image_path = UPLOAD_DIR / f"{token}{suffix}"
                gradcam_path = UPLOAD_DIR / f"{token}_gradcam.jpg"
                file.save(image_path)

                try:
                    result = predictor.predict(image_path, gradcam_path)
                    image_url = f"static/uploads/{image_path.name}"
                    gradcam_url = f"static/uploads/{gradcam_path.name}"
                except Exception as exc:
                    error = f"Prediction failed: {exc}"

        return render_template(
            "index.html",
            result=result,
            error=error,
            image_url=image_url,
            gradcam_url=gradcam_url,
        )

    return app
