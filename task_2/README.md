# Task 2: Image Segmentation in PyTorch 

This repository contains a semantic segmentation project in PyTorch, model is trained to segment images into multiple classes. It follows from **Task 1**, which involved preparing a dataset (generating masks, setting up directories).

- Model training.
- Weighted loss and sampling techniques to handle class imbalance.
- Tracking metrics via a public dashboard (TensorBoard).



## Setup
1. **Clone this repository**:
    ```bash
    git clone https://github.com/YourUsername/YourRepo.git
    ```
2. **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv vjt
    source vjt/bin/activate  # For Linux/macOS
    # or "vjt\Scripts\activate" on Windows
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation
1. **Download the Dataset and generate mask**  
   - Follow the  Task 1 README.md.
2. **Update `config.yaml`**with your data paths if needed.


## Training
1. **Configuration**  
   - edit hyperparameters in `config.yaml`:
     - Learning rate
     - Batch size
     - Number of epochs
     - Weighted loss toggles, etc. 

2. **Run the Training Script**  
   ```bash
   python3 train.py --config config.yaml

## Inferencing
1. **Run the inference Script**  
    - edit hyperparameters in `config.yaml`:
     - image paths
     - image dirs, etc.

2. **Run the inferencing Script**  
    ```bash
    python3 inference.py --config config.yaml 
