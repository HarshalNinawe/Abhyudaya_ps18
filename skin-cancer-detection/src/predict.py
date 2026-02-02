"""
Prediction utilities for skin cancer detection.
"""

import numpy as np
from tensorflow.keras.models import load_model
from src.config import BEST_MODEL_PATH
from src.dataset import load_and_preprocess_image
import logging


class SkinCancerPredictor:
    """
    Predictor class for skin cancer detection.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the saved model. If None, uses BEST_MODEL_PATH
        """
        if model_path is None:
            model_path = BEST_MODEL_PATH
        
        self.model = load_model(model_path)
        self.class_names = ['Benign', 'Malignant']
        
        logging.info(f"Model loaded from: {model_path}")
    
    def predict(self, image_path):
        """
        Predict if a skin lesion is benign or malignant.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing prediction results
        """
        # Load and preprocess image
        img_array = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Get class and confidence
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else (1 - prediction)
        
        result = {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probability_malignant': float(prediction),
            'probability_benign': float(1 - prediction)
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def predict_single_image(image_path, model_path=None):
    """
    Convenience function to predict a single image.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the saved model
        
    Returns:
        dict: Prediction results
    """
    predictor = SkinCancerPredictor(model_path=model_path)
    return predictor.predict(image_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict skin cancer from image')
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the saved model'
    )
    
    args = parser.parse_args()
    
    result = predict_single_image(args.image_path, args.model_path)
    
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probability Benign: {result['probability_benign']:.2%}")
    print(f"Probability Malignant: {result['probability_malignant']:.2%}")
    print("="*50 + "\n")
