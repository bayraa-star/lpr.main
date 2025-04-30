from fast_alpr import ALPR
from pathlib import Path

# Initialize ALPR with default models
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# Define the path to the assets directory
assets_dir = Path("assets")

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png"}

# Iterate through all files in the assets directory
for image_path in assets_dir.glob("*"):
    if image_path.suffix.lower() in image_extensions:
        try:
            # Predict on the current image
            alpr_results = alpr.predict(str(image_path))
            # Check if there are any results
            if alpr_results:
                # Extract and print the plate text from the first result
                plate_text = alpr_results[0].ocr.text
                print(plate_text)
        except Exception as e:
            # Silently skip errors
            pass
