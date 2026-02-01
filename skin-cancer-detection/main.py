"""
Main entry point for the Skin Cancer Detection project.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_single_image
from src.utils import setup_logging


def main():
    """
    Main function to run the skin cancer detection application.
    """
    parser = argparse.ArgumentParser(
        description='Skin Cancer Detection - Main Application'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--model-type',
        type=str,
        default='cnn',
        choices=['cnn', 'transfer'],
        help='Type of model to train'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the saved model'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    predict_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the saved model'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Execute command
    if args.command == 'train':
        train_model(model_type=args.model_type, epochs=args.epochs)
    
    elif args.command == 'evaluate':
        evaluate_model(model_path=args.model_path)
    
    elif args.command == 'predict':
        result = predict_single_image(args.image_path, args.model_path)
        print("\n" + "="*50)
        print("Prediction Results")
        print("="*50)
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability Benign: {result['probability_benign']:.2%}")
        print(f"Probability Malignant: {result['probability_malignant']:.2%}")
        print("="*50 + "\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
