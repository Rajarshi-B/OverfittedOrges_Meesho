import torch
import torchvision.transforms as transforms
from PIL import Image
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import timm
from torch.utils.data import DataLoader, Dataset
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight





# Initialize Swin Transformer model

train_path = '../Train_Images_CROPPED'
train_csv_path = '../CSVs/train_filled_30.csv'
output_path = '../CSVs'
threshold = 0.9 # Threshold for accuracy, to io impute



data = pd.read_csv(train_csv_path)

data['filename'] = data['id'].astype('str').str.zfill(6) + '.jpg'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
swin_model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
swin_model = swin_model.to(device)
swin_model.eval()
swin_model.reset_classifier(0)  # Remove classifier for feature extraction

# Transform for preprocessing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset to handle image loading and transformations
class ImageDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            image = torch.zeros(3, 224, 224)  # Placeholder for failed images
        return image

# Function to extract features using DataLoader
def extract_features_dataloader(filepaths, batch_size=32):
    dataset = ImageDataset(filepaths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=32, pin_memory=True)

    features_list = []
    for images in tqdm((dataloader),desc='Extracting features'):
        images = images.to(device)
        with torch.no_grad():
            features = swin_model(images).cpu().numpy()
        features_list.extend(features)
    
    return features_list

# File paths
filepaths = [os.path.join(train_path, fname) for fname in data['filename'].tolist()]

# Extract and add features to the dataframe
data['image_features'] = extract_features_dataloader(filepaths, batch_size=128)




# Concatenate image features with other attributes
attr_list = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6', 'attr_7', 'attr_8', 'attr_9', 'attr_10']
relevant_data = pd.concat([data['Category'],data[attr_list], 
                           pd.DataFrame(data['image_features'].tolist())], axis=1)


cats = relevant_data['Category'].unique()
num_of_attr = [5,10,9,8,10]
attr_for_each = [[f'attr_{i+1}' for i in range(j)] for j in num_of_attr]
attr_for_each


relevant_data_orig = relevant_data.copy(deep=True)
relevant_data_orig.head()


data_frames = {}
label_encoders = {}  # Store label encoders for each category and column

for cat, attrs in tqdm(zip(cats, attr_for_each),desc='Imputing'):
    relevant_data = relevant_data_orig[relevant_data_orig['Category'] == cat].copy(deep=True)
    relevant_data.columns = relevant_data.columns.astype(str)
    # Encode categorical data, handling NaN values carefully
    label_encoders[cat] = {}
    dont_impute_columns = []
    for column in tqdm(attrs):
        le = LabelEncoder()
        # Temporarily fill NaNs with 'Missing' only for encoding purposes
        relevant_data[column] = relevant_data[column].fillna('Missing').astype(str)
        relevant_data[column] = le.fit_transform(relevant_data[column])
        label_encoders[cat][column] = le  # Store the encoder for later decoding
        
        # After encoding, reset 'Missing' values back to NaN for imputation
        relevant_data[column] = relevant_data[column].replace(le.transform(['Missing'])[0], np.nan)
    
    # Impute missing values using RandomForest
    for column in tqdm(attrs):
        train_data = relevant_data[relevant_data[column].notna()]
        impute_data = relevant_data[relevant_data[column].isna()]

        # Debugging: Print the number of missing values in this column
        print(f"Category: {cat}, Column: {column}, Missing values: {relevant_data[column].isna().sum()}")

        if not impute_data.empty:
            X_all = train_data.drop(columns=[column, 'Category'])
            y_all = train_data[column]
            X_impute = impute_data.drop(columns=[column, 'Category'])

            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, stratify=y_all, test_size=0.2, random_state=42)

            # Train classifier
            tree_model_rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            tree_model_rf.fit(X_train, y_train)
            y_pred_rf = tree_model_rf.predict(X_val)
            acc = accuracy_score(y_val, y_pred_rf)
            f1 = f1_score(y_val,y_pred_rf, average='weighted')
            # print(f'{column} - Random Forest Accuracy: {acc}')

            # Impute missing values
            if(acc>threshold):
                print(f'{column} accuracy_score {acc} > {threshold} : will be imputed, f1 = {f1}')
                imputed_values = tree_model_rf.predict(X_impute)
                relevant_data.loc[relevant_data[column].isna(), column] = imputed_values
            else:
                dont_impute_columns.append(column)
                print(f'{column} accuracy_score {acc} < {threshold} : will not be imputed, f1 = {f1}')
                val_to_fill = [i for i,j in enumerate(label_encoders[cat][column].classes_) if j=='Missing'][0]
                relevant_data.loc[relevant_data[column].isna(), column] = val_to_fill
    
    # Decode labels back to original and replace 'Missing' with NaN
    for column in attrs:
        # if(column not in dont_impute_columns):
        le = label_encoders[cat][column]
        relevant_data[column] = le.inverse_transform(relevant_data[column].astype(int))
        relevant_data[column] = relevant_data[column].replace('Missing', np.nan)
        
    
    # Store the processed data
    data_frames[cat] = relevant_data
    print(f"Finished processing category: {cat}\n")


# Concatenate all dataframes in the dictionary into a single dataframe
combined_df = pd.concat(data_frames.values(), ignore_index=True)

# Check the resulting combined dataframe
print(combined_df.shape)  # To see the total rows and columns

combined_df.head()

combined_df.to_parquet(f'{output_path}/train_45k_imputed_with_{threshold}_threshold_test.parquet',engine='fastparquet')
combined_df.insert(0,'id', data['id'].values)
combined_df.insert(2,'len', data['len'].values)
combined_df.drop(columns=combined_df.columns[13:]).to_csv(f'{output_path}/train_45k_imputed_with_{threshold}_threshold_test.csv',index=False)