import cv2
from fast_alpr import ALPR
from pathlib import Path

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
    )

assests_dir = Path("assets")
image_extensions = {".jpg", ".jpeg", ".png"}

for image_path in assests_dir.glob("*"):
    if image_path.suffix.lower() in image_extensions:
        print(f"Processing {image_path}...")
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                img_resized = cv2.resize(img, (384, 384))
                alpr_results = alpr.predict(img_resized)
                print(f"Results for {image_path}: {alpr_results}")
            else:
                print(f"Could not load {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")