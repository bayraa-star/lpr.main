import logging
from fast_alpr import ALPR
from pathlib import Path
from custom_ocr import CustomOCR
from custom_detector import YOLOv5Detector
import traceback

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Test CustomOCR initialization
try:
    logger.debug("Attempting to initialize CustomOCR with model 'mongolian_plate_ocr.onnx'")
    custom_ocr = CustomOCR("mongolian_plate_ocr.onnx")
    logger.info("CustomOCR initialized successfully.")
except Exception as e:
    logger.error(f"CustomOCR initialization failed: {e}\n{traceback.format_exc()}")
    raise

# Test ALPR initialization
try:
    logger.debug("Attempting to initialize YOLOv5Detector with model '/home/crossroad/Documents/Videos/screenshots/yolov5/runs/train/exp3/weights/best.pt'")
    custom_detector = YOLOv5Detector("/home/crossroad/Documents/Videos/screenshots/yolov5/runs/train/exp3/weights/best.pt")

    logger.debug("Attempting to initialize ALPR with custom YOLOv5 detector")
    alpr = ALPR(
        detector=custom_detector,
        ocr=custom_ocr,
        save_cropped_plates=True,
        cropped_plates_dir="cropped_plates"
    )
    logger.info("ALPR initialized successfully.")
except Exception as e:
    logger.error(f"ALPR initialization failed: {e}\n{traceback.format_exc()}")
    raise

# Define assets directory
assets_dir = Path("assets")
if not assets_dir.exists():
    logger.error(f"Assets directory '{assets_dir}' does not exist.")
    raise FileNotFoundError(f"Directory '{assets_dir}' not found")

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png"}

# Process images in assets directory
for image_path in assets_dir.glob("*"):
    if image_path.suffix.lower() in image_extensions:
        logger.info(f"Starting processing for image: {image_path}")
        try:
            # Verify image file accessibility
            if not image_path.is_file():
                logger.error(f"{image_path} is not a valid file.")
                continue
            
            logger.debug(f"Calling ALPR.predict on {image_path}")
            alpr_results = alpr.predict(str(image_path))
            
            # Check if results are returned
            if alpr_results is None:
                logger.warning(f"ALPR.predict returned None for {image_path}")
                continue
            if not alpr_results:
                logger.warning(f"No detections returned by ALPR for {image_path}")
                continue

            logger.debug(f"Processing {len(alpr_results)} detection(s) for {image_path}")
            for idx, result in enumerate(alpr_results):
                # Validate result structure
                if not hasattr(result, 'ocr'):
                    logger.error(f"Detection {idx} in {image_path} has no 'ocr' attribute")
                    continue
                
                if result.ocr is None:
                    logger.warning(f"OCR result is None for detection {idx} in {image_path}")
                    continue
                
                if not hasattr(result.ocr, 'text'):
                    logger.error(f"OCR object for detection {idx} in {image_path} has no 'text' attribute")
                    continue
                
                if result.ocr.text:
                    plate_text = result.ocr.text
                    logger.info(f"Detected plate text: '{plate_text}' for detection {idx} in {image_path}")
                    print(plate_text)
                else:
                    logger.warning(f"Empty or no OCR text for detection {idx} in {image_path}")
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}\n{traceback.format_exc()}")
            continue