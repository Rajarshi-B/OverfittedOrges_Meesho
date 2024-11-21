import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import joblib
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_csv = pd.read_csv('../CSVs/test.csv')
path_test = '../Test_Images_CROPPED/test_images_cropped'       #'/home/ravindra/meesho/test_images_cropped'
csv_name_submission = '../CSVs/Submission_batch32_predictions50.csv'
path_to_folder = '../MODELS&PICKLES/MODEL_WEIGHTS/ResNet_50_weights'
path_to_le_obj = '../MODELS&PICKLES/PICKLES/pickels_data_FOR_TRAINING_cropped'
# Assuming test_csv and Categories are predefined
Category_list = ['Men Tshirts','Sarees','Kurtis','Women Tshirts','Women Tops & Tunics']




test_csv['filename'] = test_csv['id'].copy(deep=True)
for i in tqdm(range(len(test_csv)),desc='Filename column'):
    test_csv.loc[i,'filename'] = str(int(test_csv['id'].iloc[i]) + 1000000)[1:] + '.jpg'
# len_list = [5,10,9,8,10]
test_csv['len'] = [5]*len(test_csv)
len_dict = {'Men Tshirts':5, 'Sarees':10, 'Kurtis':9, 'Women Tshirts':8, 'Women Tops & Tunics':10}
# for key in len_dict.keys():
#     test_csv[test_csv['Category'] == key]['len'].iloc[:] = [len_dict[key]]*len(test_csv[test_csv['Category'] == key]['len'].iloc[:])
for i in tqdm(range(len(test_csv)),desc='len column'):
    test_csv.loc[i,'len'] = len_dict[test_csv['Category'].iloc[i]]

nu_list = ['nu']*len(test_csv)
for i in range(1,11):
    test_csv[f'attr_{i}'] = nu_list
test_csv.head()


Categories = list(test_csv['Category'].unique())



# Function to load an image and resize it
def load_image(file_path):
    return np.asarray(Image.open(file_path).resize((224, 224)))

# Dataset class for loading images efficiently
class ImageDataset(Dataset):
    def __init__(self, filenames, path_to, transform=None):
        self.filenames = filenames
        self.path_to = path_to
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = f"{self.path_to}/{self.filenames[idx]}"
        image = load_image(img_path)
        image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        return image

# Custom PyTorch module for the building classifier with optional dropout
class BuildingClassifierWithDropout(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.5, unfreeze_layers=1, resnet_version=50):
        super(BuildingClassifierWithDropout, self).__init__()
        # Select the appropriate ResNet version
        if resnet_version == 101:
            self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        elif resnet_version == 18:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_version == 34:
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet_version == 50:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all layers by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layers based on the unfreeze_layers parameter
        if unfreeze_layers >= 1:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 2:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 3:
            for param in self.model.layer3.parameters():
                param.requires_grad = True

        # Replace the fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dropout_fc = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout_fc(x)
        return x

# Function to make predictions using the model
def model_predict(test_loader, model):
    model_preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images in test_loader:  # Images are already batched
            images = images.to(device)  # Move images to the device
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            model_preds.extend(preds.cpu().numpy())
    return model_preds

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model and label encoder object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for i in tqdm(Category_list, desc='Category'):
    if i=='Men Tshirts':
        z=5
    elif i=='Sarees':
        z=10
    elif i=='Kurtis':
        z=9
    elif i=='Women Tshirts':
        z=8
    elif i=='Women Tops & Tunics':
        z=10

    for j in tqdm([f'attr_{k}' for k in range(1,z+1)], desc=f'{i} attribute'):
        attr_name = j
        remark = f'{i}_{attr_name}'
        le_name = [j for j in os.listdir(path_to_le_obj) if f'{attr_name}_{i}' in j and 'Class_names' in j][0]
        le_obj = joblib.load(f'{path_to_le_obj}/{le_name}')
        model_name = [i for i in os.listdir(path_to_folder) if remark in i][0]
        model = BuildingClassifierWithDropout(num_classes=len(le_obj), dropout_prob=0.5, unfreeze_layers=2, resnet_version=50)
        model.load_state_dict(torch.load(os.path.join(path_to_folder, model_name), map_location=device,weights_only = True))
        # print(le_obj)
        # print(remark)
        filename_list = list(test_csv[test_csv['Category'] == i]['filename'].values)
        test_loader = DataLoader(ImageDataset(filename_list, path_test, transform), batch_size=32, shuffle=False,num_workers=32)
        predictions = model_predict(test_loader, model)
        preds_for_this_category_and_attribute = [le_obj[i] for i in predictions]
        # print(len(test_csv.loc[test_csv['Category'] == i, attr_name]),len(preds_for_this_category_and_attribute))
        # print(preds_for_this_category_and_attribute)
        test_csv.loc[test_csv['Category'] == i, attr_name] = preds_for_this_category_and_attribute

# Output the predictions to CSV or further processing


(test_csv.drop(columns='filename')).to_csv(csv_name_submission,index=False)