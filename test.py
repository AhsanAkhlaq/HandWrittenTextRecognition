import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import ndimage
import string

class WordRecognizer:
    def __init__(self, model_path):
        """
        Initialize the word recognizer with your trained character model
        """
        self.model = keras.models.load_model(model_path)
        
        # EMNIST ByClass mapping (0-9, A-Z, a-z)
        self.char_mapping = self._create_emnist_mapping()
        
    def _create_emnist_mapping(self):
        """Create mapping from model output indices to characters"""
        chars = []
        # Digits 0-9
        chars.extend([str(i) for i in range(10)])
        # Uppercase A-Z
        chars.extend(list(string.ascii_uppercase))
        # Lowercase a-z
        chars.extend(list(string.ascii_lowercase))
        return chars
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input word image
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply binary threshold
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return img
    
    def segment_characters(self, img):
        """
        Segment individual characters from word image
        """
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        character_images = []
        bounding_boxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours (noise)
            if w < 10 or h < 10:
                continue
            
            # Extract character region
            char_img = img[y:y+h, x:x+w]
            char_img = cv2.flip(char_img, 1)
            char_img = cv2.rotate(char_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            character_images.append(char_img)
            bounding_boxes.append((x, y, w, h))
        
        return character_images, bounding_boxes
    
    def prepare_character_for_model(self, char_img):
        """
        Prepare individual character image for your trained model
        """
        # Resize to 28x28 (EMNIST standard)
        char_img = cv2.resize(char_img, (28, 28))
        
        # Normalize pixel values
        char_img = char_img.astype('float32') / 255.0
        
        # Add batch dimension
        char_img = char_img.reshape(1, 28, 28, 1)
        
        return char_img
    
    def predict_character(self, char_img):
        """
        Predict single character using your trained model
        """
        # Prepare image for model
        processed_char = self.prepare_character_for_model(char_img)
        
        # Get prediction
        prediction = self.model.predict(processed_char, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Map to character
        predicted_char = self.char_mapping[predicted_class]
        
        return predicted_char, confidence
    
    def recognize_word(self, image_path, min_confidence=0.5):
        """
        Main function to recognize word from image
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Segment characters
        char_images, bounding_boxes = self.segment_characters(img)
        
        if not char_images:
            return "", [], []
        
        # Predict each character
        predicted_chars = []
        confidences = []
        
        for char_img in char_images:
            char, conf = self.predict_character(char_img)
            
            # Only include predictions above confidence threshold
            if conf >= min_confidence:
                predicted_chars.append(char)
                confidences.append(conf)
            else:
                predicted_chars.append('?')  # Unknown character
                confidences.append(conf)
        
        # Combine characters to form word
        recognized_word = ''.join(predicted_chars)
        
        return recognized_word, confidences, bounding_boxes
    
    def visualize_recognition(self, image_path):
        """
        Visualize the recognition process
        """
        # Load original image
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        preprocessed_img = self.preprocess_image(image_path)
        
        # Get recognition results
        word, confidences, bboxes = self.recognize_word(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original image
        axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(preprocessed_img, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')
        
        # Character segmentation
        img_with_boxes = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
        for i, (x, y, w, h) in enumerate(bboxes):
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_with_boxes, f'{i+1}', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[1, 0].imshow(img_with_boxes)
        axes[1, 0].set_title('Character Segmentation')
        axes[1, 0].axis('off')
        
        # Results
        result_text = f"Recognized Word: '{word}'\n\n"
        result_text += "Character Confidences:\n"
        for i, (char, conf) in enumerate(zip(word, confidences)):
            result_text += f"Char {i+1}: '{char}' ({conf:.3f})\n"
        
        axes[1, 1].text(0.1, 0.5, result_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Recognition Results')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return word, confidences

# Example usage
def main():
    # Initialize word recognizer with your trained model
    recognizer = WordRecognizer('D:\\Workspace\\PYTHON\\ML_Models\\HandWrittenTextRecognition\\model\\emnist_cnn_best.h5')
    
    # Recognize word from image
    image_path = 'D:\\Workspace\\PYTHON\\ML_Models\\HandWrittenTextRecognition\\images\\sample1.jpg'

    # Simple recognition
    word, confidences, bboxes = recognizer.recognize_word(image_path)
    print(f"Recognized word: '{word}'")
    print(f"Character confidences: {confidences}")
    
    # Visualize the process
    recognizer.visualize_recognition(image_path)

if __name__ == "__main__":
    main()

# Additional utility functions for batch processing
class BatchWordRecognizer(WordRecognizer):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def recognize_multiple_words(self, image_paths):
        """
        Recognize multiple word images
        """
        results = []
        for img_path in image_paths:
            try:
                word, confidences, bboxes = self.recognize_word(img_path)
                results.append({
                    'image_path': img_path,
                    'recognized_word': word,
                    'confidences': confidences,
                    'avg_confidence': np.mean(confidences) if confidences else 0
                })
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'recognized_word': '',
                    'error': str(e)
                })
        return results
    
    def save_results_to_csv(self, results, output_path):
        """
        Save recognition results to CSV
        """
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

# Enhanced preprocessing for better accuracy
class EnhancedWordRecognizer(WordRecognizer):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def advanced_preprocess(self, image_path):
        """
        Advanced preprocessing techniques
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Noise reduction
        img = cv2.medianBlur(img, 3)
        
        # Contrast enhancement
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive threshold for varying lighting
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        return img
    
    def improved_segmentation(self, img):
        """
        Improved character segmentation
        """
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        character_images = []
        bounding_boxes = []
        
        # Calculate average character dimensions for filtering
        if contours:
            heights = [cv2.boundingRect(c)[3] for c in contours]
            avg_height = np.mean(heights)
            min_height = avg_height * 0.3
            max_height = avg_height * 2.0
        else:
            min_height, max_height = 10, 1000
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Improved filtering
            if w < 5 or h < min_height or h > max_height:
                continue
            
            # Add padding around character
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2*padding)
            h = min(img.shape[0] - y, h + 2*padding)
            
            char_img = img[y:y+h, x:x+w]
            character_images.append(char_img)
            bounding_boxes.append((x, y, w, h))
        
        return character_images, bounding_boxes