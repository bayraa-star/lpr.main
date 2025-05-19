import os
import cv2
import numpy as np
import onnxruntime as ort
from fast_alpr.base import BaseOCR, OcrResult
import datetime

class CustomOCR(BaseOCR):
    def __init__(self, model_path: str):
        """Initialize the custom OCR with a local ONNX model file."""
        self.model = ort.InferenceSession(model_path)
        self.alphabet = "0123456789АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ-"
        self.pad_char = "-"
        self.max_plate_slots = 10
        self.img_height = 64
        self.img_width = 128
        # Directory to save preprocessed images
        self.save_dir = "processed_ocr_detection"
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self, image: np.ndarray, original_filename: str = "unknown") -> OcrResult:
        """Run OCR prediction on the input image and save the preprocessed image."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Preprocess the image: resize, convert to grayscale, normalize
        image = cv2.resize(image, (self.img_width, self.img_height))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        
        # Save the preprocessed image
        image_for_save = (image * 255).astype(np.uint8)  # Convert back to uint8 for saving
        save_path = os.path.join(self.save_dir, f"preprocessed_{original_filename}_{timestamp}.png")
        cv2.imwrite(save_path, image_for_save)

        # Prepare the image for inference
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Get input and output names
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        # Run inference
        prediction = self.model.run([output_name], {input_name: image})[0]

        # Decode the prediction
        plate_text, confidence = self._process_output(prediction)

        return OcrResult(text=plate_text, confidence=confidence)

    def _process_output(self, prediction):
        """Convert model output to a readable plate string and confidence score."""
        # Assuming prediction shape: (1, max_plate_slots, vocabulary_size)
        pred_probs = np.max(prediction, axis=2)[0]  # Max probabilities for each slot
        pred_indices = np.argmax(prediction, axis=2)[0]  # Indices of max probabilities

        plate_text = ""
        for idx in pred_indices:
            char = self.alphabet[idx]
            if char != self.pad_char:
                plate_text += char

        # Calculate average confidence across non-padding characters
        confidence_scores = [pred_probs[i] for i, char in enumerate(plate_text) if char != self.pad_char]
        confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return plate_text, confidence