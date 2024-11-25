### README for Project Repository

#### Project Structure Overview
The repository is structured to facilitate the data preprocessing, model training, and prediction submission phases of our machine learning project. Below is a detailed breakdown of the directory structure:

```
.
├── CSVs                      # Contains various CSV and Parquet files for training and testing
├── MODELS&PICKLES            # Storage for trained model weights and pickle files
│   ├── MODEL_WEIGHTS         # PyTorch model weights for ResNet50 across different attributes and categories
│   └── PICKLES               # Data files saved in pickle format, used during training
├── PREPROCESSING             # Scripts and configurations for data preprocessing
│   ├── 1_Hashing_Remove_DuplicatesUsingHashes.py  # Script for removing duplicate images using hashing
│   ├── 2_Representative_Images_CLIP.py            # Script for selecting representative images using CLIP and k-means
│   ├── 3_YOLO                                      # YOLO configurations and scripts for image cropping
│   └── 4_Swin_RF_Imputation.py                     # Script for imputing missing CSV data using SWIN transformer and Random forest
├── representative_images    # Folder for storing representative images
├── SUBMISSION               # Contains scripts for generating submission files
├── Test_Images_CROPPED      # Cropped test images
├── TRAIN                    # Training scripts and notebooks
│   ├── Exploration_Multihead_ResNET-XGB           # Notebook for multihead model experiments
│   └── train_ResNET                                # Main training script for ResNet models
├── train_images             # Original training images
└── Train_Images_CROPPED     # Cropped training images
```

#### Data Preprocessing
1. **Duplicate Removal:** Utilize the `1_Hashing_Remove_DuplicatesUsingHashes.py` script to remove duplicates from the dataset, ensuring data uniqueness.
2. **Representative Images:** Apply the `2_Representative_Images_CLIP.py` to select a subset of representative images using the CLIP model combined with k-means clustering.
3. **Cropping:** Use the configurations and scripts within the `3_YOLO` directory to crop images based on detected objects, enhancing the model's focus on relevant features.
4. **CSV Imputation:** The `4_Swin_RF_Imputation.py` script is used to fill in missing data in CSV files, combining SWIN transformers with Random Forest algorithms to ensure comprehensive data for training.

#### Training
1. **ResNet Training:** Run the `train_RESNET_MAIN.py` script located in the `TRAIN/train_ResNET` directory to train the ResNet models on the processed data.
2. **Advanced Experiments:** For further experimentation, explore the `Exploration_Multihead_ResNET-XGB` notebook which integrates ResNet with XGBoost in a multi-head configuration to tackle multiple attributes simultaneously.

#### Submission
- To generate the submission CSV files, execute the `submission_All_ResNet.py` script in the `SUBMISSION` directory. This script compiles the predictions from the trained ResNet models into the required format.

### How to Use This Repository
1. Ensure all dependencies are installed as per the requirements.txt (not shown but assumed).
2. Follow the steps in the preprocessing scripts to prepare your data.
3. Train the models using the provided scripts and Jupyter notebooks.
4. Generate your submission files using the `SUBMISSION` script.
5. Review the `README.md` files within subdirectories for specific instructions on scripts and configurations.
