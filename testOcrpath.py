from fast_alpr import ALPR
from custom_ocr import CustomOCR

custom_ocr = CustomOCR("mongolian_plate_ocr.onnx")
# Initialize ALPR with a custom OCR model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr=custom_ocr
)

# Print the OCR model path
print(f"Using OCR model: {alpr.ocr}")