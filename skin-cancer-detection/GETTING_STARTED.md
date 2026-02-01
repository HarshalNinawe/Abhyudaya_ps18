# Getting Started with Skin Cancer Detection 🚀

This guide will help you quickly set up and start using the Skin Cancer Detection project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Step-by-Step Setup

### 1. Navigate to the Project Directory

```bash
cd skin-cancer-detection
```

### 2. Run the Setup Script

#### For Linux/Mac:
```bash
./setup.sh
```

#### For Windows:
```bash
setup.bat
```

#### Or Manual Setup:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Organize your skin cancer images in the following structure:

```
data/raw/
├── train/
│   ├── benign/       # Put benign training images here
│   └── malignant/    # Put malignant training images here
├── val/
│   ├── benign/       # Put benign validation images here
│   └── malignant/    # Put malignant validation images here
└── test/
    ├── benign/       # Put benign test images here
    └── malignant/    # Put malignant test images here
```

**Recommended Split:**
- Training: 70-80% of your data
- Validation: 10-15% of your data
- Testing: 10-15% of your data

## Quick Start Commands

### 1. Explore the Data (Optional)

Open Jupyter Notebook to explore your dataset:

```bash
jupyter notebook
```

Navigate to `notebooks/data_exploration.ipynb`

### 2. Train a Model

Train a simple CNN model:

```bash
python main.py train --model-type cnn --epochs 50
```

Or train a transfer learning model (recommended for better accuracy):

```bash
python main.py train --model-type transfer --epochs 50
```

### 3. Monitor Training

In a new terminal, start TensorBoard to monitor training:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

### 4. Evaluate the Model

After training, evaluate your model:

```bash
python main.py evaluate
```

### 5. Make Predictions

Predict on a new image:

```bash
python main.py predict path/to/your/image.jpg
```

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 2. Check if data is properly organized
ls data/raw/train/benign/
ls data/raw/train/malignant/

# 3. Train the model
python main.py train --model-type transfer --epochs 30

# 4. In another terminal, monitor training
tensorboard --logdir logs/tensorboard

# 5. After training, evaluate
python main.py evaluate

# 6. Make a prediction
python main.py predict data/raw/test/benign/test_image.jpg
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

Make sure you activated the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Then reinstall dependencies:
```bash
pip install -r requirements.txt
```

### "No such file or directory: data/raw/train"

Make sure you've organized your dataset in the correct structure. See step 3 above.

### Training is very slow

- Reduce batch size in `src/config.py`
- Use a smaller number of epochs for testing
- Consider using a GPU if available

### Out of Memory errors

- Reduce batch size in `src/config.py` (e.g., from 32 to 16 or 8)
- Use a smaller image size
- Close other applications to free up RAM

## Next Steps

1. **Experiment with hyperparameters** - Edit `src/config.py` to try different:
   - Learning rates
   - Batch sizes
   - Image sizes
   - Data augmentation parameters

2. **Try the notebooks** - Use Jupyter notebooks for interactive experimentation:
   - `notebooks/data_exploration.ipynb` - Explore your dataset
   - `notebooks/model_experiments.ipynb` - Experiment with models

3. **Customize the model** - Edit `src/model.py` to try different architectures

4. **Add more data** - More training data typically improves model performance

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments in the `src/` directory
- Check the logs in `logs/training.log` for error messages

## Dataset Resources

If you need a skin cancer dataset, consider these sources:
- [ISIC Archive](https://www.isic-archive.com/)
- [Kaggle Skin Cancer Dataset](https://www.kaggle.com/datasets)
- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

**Important**: Always ensure you have the right to use any dataset and follow ethical guidelines for medical AI applications.

---

Happy coding! 🎉
