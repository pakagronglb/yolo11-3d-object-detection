# Troubleshooting Guide

## Issues Identified

1. **NumPy Version Compatibility**
   - The project was using NumPy 2.x which is causing compatibility issues with some modules
   - Error message: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.1 as it may crash`

2. **Hugging Face Authentication**
   - The Hugging Face token in the `.env` file wasn't being loaded properly
   - Error when trying to download the depth model

3. **YOLO Model Compatibility**
   - The YOLO model file (yolo11n.pt) is not compatible with the installed ultralytics version
   - Error: `Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'`

## Solutions

### 1. NumPy Compatibility Fix

```bash
# Downgrade NumPy to a compatible version
pip uninstall -y numpy
pip install numpy==1.24.3
```

### 2. Hugging Face Authentication Fix

1. Add the dotenv import and loading to depth_model.py:

```python
from dotenv import load_dotenv

# In the DepthEstimator.__init__ method:
load_dotenv()  # Load environment variables including HF_TOKEN
```

2. Make sure the .env file contains your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token
```

3. Install python-dotenv:

```bash
pip install python-dotenv
```

### 3. YOLO Model Compatibility Fix

The YOLO model file (yolo11n.pt) was created with a specific version of ultralytics that includes the 'C3k2' module. There are two potential solutions:

1. **Install the exact ultralytics version that the model was created with**:
   - This might require trying different versions until finding the compatible one
   - Try: `pip install ultralytics==8.0.0` or other versions

2. **Download a compatible model file**:
   - Get a YOLO model file that's compatible with the installed ultralytics version
   - Standard YOLOv8 models should work with most ultralytics versions

## Complete Environment Setup

For a clean setup, you can use the following commands:

```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install compatible versions of key packages
pip install numpy==1.24.3
pip install ultralytics==8.0.0
pip install transformers==4.38.0
pip install python-dotenv==1.0.0
pip install -r requirements.txt

# Run the application
python run.py
```

## Additional Debugging

If you continue to experience issues, you can run the debug scripts:

```bash
python debug_models.py  # Test both models separately
python fix_environment.py  # Fix environment issues
``` 