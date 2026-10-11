import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
NEGATIVE_NAMES = {
    "no", "notumor", "notumour", "nontumor", "nontumour",
    "normal", "healthy", "negative"
}
POSITIVE_NAMES = {
    "yes", "tumor", "tumour", "positive", "glioma", "gliomatumor",
    "gliomatumour", "meningioma", "meningiomatumor",
    "meningiomatumour", "pituitary", "pituitarytumor", "pituitarytumour"
}

def normalize_name(value):
    return str(value).lower().replace(" ", "").replace("_", "").replace("-", "")

def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS

def get_all_images(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and is_image(p)]

def count_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and is_image(p))

def detect_binary_label(path):
    parts = [normalize_name(x) for x in Path(path).parent.parts]
    for part in parts:
        if part in NEGATIVE_NAMES:
            return "no_tumor"
    for part in parts:
        if part in POSITIVE_NAMES:
            return "tumor"
    return None

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
