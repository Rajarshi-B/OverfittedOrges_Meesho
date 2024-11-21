# YOLO Clothing Detector

This guide explains how to set up and run a YOLO model for detecting clothing in images. The project is structured into two main scripts: training the model (`train.py`) and cropping detected clothing items (`processing_cropping.py`).

## Directory Structure

Ensure your data is organized as follows:
- **Folder Structure**:
    ```
    Preprocessing_MLCPS/
    ├── yolo_A/
    │   ├── config.yaml            # YOLO configuration file for class A
    │   ├── train.py            #for training YOLO on class A data
    │   ├── data/
    │   │   ├── images/
    │   │   │   ├── train/         # Training images for class A
    │   │   │   └── val/           # Validation images for class A
    │   │   ├── labels/
    │   │   │   ├── train/         # Corresponding labels (.txt files) for training images
    │   │   │   └── val/           # Corresponding labels (.txt files) for validation images
    ```


## Configuration File Setup

1. **Config File**: The `config.yaml` file specifies the paths to the training and validation data. You need to update this file to reflect the paths on your system. Here's an example structure of the `config.yaml` file:

    ```yaml
    train: data/images/train
    val: data/images/val

    names: ['class1', 'class2', ...]
    ```

2. **Edit the File**:list the class names under `names`.

## Training the Model

1. **Script**: The `train.py` script is used to train the YOLO model.

2. **Usage**:
    ```bash
    python train.py
    ```
    - Specify the path to the `config.yaml` file.
    - Change the number of training epochs.
    - The initial weights (pretrained model).

3. **Outputs**: The trained models and output logs will be saved in the `runs/` folder.

## Cropping Detected Clothing

1. **Script**: Use the `processing_cropping.py` script to crop images based on detected clothing items.

2. **Setup**:
    - Specify the path to the fine-tuned model weights. Typically, these are stored in `YOLO/runs/weights/best.pt`.
    - Define the input directory containing the images you want to process.
    - Specify the output directory where the cropped images will be saved.

3. **Usage**:
    ```bash
    python processing_cropping.py
    ```

Ensure you have the required dependencies installed, including PyTorch, NumPy, and the Ultralytics YOLO package for Python. Follow the installation guide provided by YOLO for detailed instructions on setting up your environment.

Follow the steps above to train your model and process images for clothing detection.
