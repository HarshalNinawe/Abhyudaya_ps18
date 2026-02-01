# Skin Cancer Detection 🔬

A deep learning project for detecting skin cancer (benign vs malignant) from images using TensorFlow/Keras.

## 📁 Project Structure

```
skin-cancer-detection/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   ├── val/
│   │   └── test/
│   └── processed/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # Model architectures
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation
│   ├── predict.py         # Prediction utilities
│   └── utils.py           # Helper functions
│
├── models/
│   ├── best_model.h5      # Saved best model
│   └── checkpoints/       # Training checkpoints
│
├── logs/
│   ├── training.log       # Training logs
│   └── tensorboard/       # TensorBoard logs
│
├── requirements.txt
├── README.md
└── main.py               # Main entry point
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd skin-cancer-detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Place your skin cancer images in the following structure:

```
data/raw/train/benign/       # Training benign images
data/raw/train/malignant/    # Training malignant images
data/raw/val/benign/         # Validation benign images
data/raw/val/malignant/      # Validation malignant images
data/raw/test/benign/        # Test benign images
data/raw/test/malignant/     # Test malignant images
```

## 📊 Usage

### Training

Train a model using the command-line interface:

```bash
# Train custom CNN model
python main.py train --model-type cnn --epochs 50

# Train transfer learning model (MobileNetV2)
python main.py train --model-type transfer --epochs 50
```

Or use the training script directly:

```bash
python src/train.py --model-type cnn --epochs 50
```

### Evaluation

Evaluate a trained model on the test set:

```bash
python main.py evaluate

# Or with a specific model
python main.py evaluate --model-path models/best_model.h5
```

### Prediction

Make predictions on new images:

```bash
python main.py predict path/to/image.jpg

# Or with a specific model
python main.py predict path/to/image.jpg --model-path models/best_model.h5
```

## 🔧 Configuration

All configuration parameters are in `src/config.py`:

- **Image size**: 224x224 pixels
- **Batch size**: 32
- **Learning rate**: 0.001
- **Number of epochs**: 50
- **Data augmentation parameters**

Modify these parameters as needed for your experiments.

## 📓 Jupyter Notebooks

The project includes two Jupyter notebooks for interactive development:

1. **data_exploration.ipynb**: Explore and visualize the dataset
2. **model_experiments.ipynb**: Experiment with different architectures

To use the notebooks:

```bash
jupyter notebook
```

## 🧠 Models

### Custom CNN Model
A custom convolutional neural network with:
- 3 convolutional blocks with batch normalization
- Max pooling layers
- Dense layers with dropout
- Binary classification output

### Transfer Learning Model
Uses MobileNetV2 pre-trained on ImageNet:
- Frozen base model layers
- Custom classification head
- Option to fine-tune top layers

## 📈 Monitoring

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

### Training Logs

Check `logs/training.log` for detailed training information.

## 🔍 Model Evaluation

The evaluation script provides:
- Test accuracy and loss
- AUC-ROC score
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- ROC curve visualization

## 📦 Requirements

Main dependencies:
- TensorFlow >= 2.10.0
- Keras >= 2.10.0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV

See `requirements.txt` for the complete list.

## 🛠️ Development

### Adding New Features

1. Add new model architectures in `src/model.py`
2. Add preprocessing functions in `src/dataset.py`
3. Add utility functions in `src/utils.py`
4. Update configuration in `src/config.py`

### Running Tests

```bash
# Add tests as the project grows
python -m pytest tests/
```

## 📝 License

This project is open source and available under the MIT License.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset: [Add your dataset source]
- Inspired by medical AI research in dermatology

---

**Note**: This is a development project. For medical applications, consult with healthcare professionals and ensure proper validation.
