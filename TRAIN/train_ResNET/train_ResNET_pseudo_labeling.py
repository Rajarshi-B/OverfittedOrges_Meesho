import time
import joblib
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

psuedo_labeling = False

num_epochs = 5
lr = 1e-3
unfreeze_layers = 2
step_size = 9
gamma = 0.1
dropout_prob = 0.25
model_save_path = ''
tag = 'testpseudoSimpleCode'
test_size = 0.2
confidence_threshold = 0.9  # Set a confidence threshold
num_workers = 16

Top_directory = f'{tag}_RESNET_MODELS'

model_folder = 'Models'
accuracy_folder = 'Accuracy_curve'
confusion_folder = 'Confusion_matrix'

try:
    os.mkdir(f'../MODELS/{Top_directory}')
except:
    print('CHANGE THE FOLDER TAG AND RUN AGAIN!')
    exit()

model_save_path = f'../MODELS/{Top_directory}/{model_folder}'
accuracy_save_path = f'../MODELS/{Top_directory}/{accuracy_folder}'
confusion_save_path = f'../MODELS/{Top_directory}/{confusion_folder}'


try:
    os.mkdir(model_save_path)
    os.mkdir(accuracy_save_path)
    os.mkdir(confusion_save_path)
except:
    print('CANNOT MAKE SUBFOLDERS!')
    exit()




############ Preprocessing

#loading data 
path = '../cropped_30k_images'  ### path to images

# Path to resnet weights.
path_to_res = '../MODELS/duplicate_removed_cropped_Run_RESNET[50]_OverSamplingFalse_unfreeze[3]_RESNET_MODELS/Models'



filenames = [filen for filen in os.listdir(f"{path}") if 'jpg' in filen]

Attribute_list = pd.read_parquet('..CSVs/category_attributes.parquet')

train_atributes = pd.read_csv('..CSVs/train_filled_30.csv')

#categories name
class_names = train_atributes['Category'].unique()




#filling nan to missing values
filename_classes = {}
for i in range(len(class_names)):
    filename_classes[class_names[i]] = (train_atributes[train_atributes['Category']==class_names[i]]['id'].values).astype('int32')


train_men_tshirt = train_atributes[train_atributes['Category']==class_names[0]].copy(deep = True)
train_sarees = train_atributes[train_atributes['Category']==class_names[1]].copy(deep = True)
train_kurtis = train_atributes[train_atributes['Category']==class_names[2]].copy(deep = True)
train_women_tshirt = train_atributes[train_atributes['Category']==class_names[3]].copy(deep = True)
train_women_top_tunics = train_atributes[train_atributes['Category']==class_names[4]].copy(deep = True)




# Drop columns which has only nan values, i.e unique() has length 1 and the item is 'nan'
for i in range(10):
    if(len(train_men_tshirt[f'attr_{i+1}'].unique())==1):
        train_men_tshirt.drop(columns=[f'attr_{i+1}'],inplace=True)
    if(len(train_sarees[f'attr_{i+1}'].unique())==1):
        train_sarees.drop(columns=[f'attr_{i+1}'],inplace=True)
    if(len(train_kurtis[f'attr_{i+1}'].unique())==1):
        train_kurtis.drop(columns=[f'attr_{i+1}'],inplace=True)
    if(len(train_women_tshirt[f'attr_{i+1}'].unique())==1):
        train_women_tshirt.drop(columns=[f'attr_{i+1}'],inplace=True)
    if(len(train_women_top_tunics[f'attr_{i+1}'].unique())==1):
        train_women_top_tunics.drop(columns=[f'attr_{i+1}'],inplace=True)





#adding extra column named filenames to each images with rename as 000000.jpg
train_men_tshirt['filename'] = train_men_tshirt['id'].astype(str).copy(deep=True)
for i in train_men_tshirt.index:
    train_men_tshirt.loc[i,'filename'] = (str(1000000+train_men_tshirt.loc[i,'id'])+'.jpg')[1:]

train_sarees['filename'] = train_sarees['id'].astype(str).copy(deep=True)
for i in train_sarees.index:
    train_sarees.loc[i,'filename'] = (str(1000000+train_sarees.loc[i,'id'])+'.jpg')[1:]

train_kurtis['filename'] = train_kurtis['id'].astype(str).copy(deep=True)
for i in train_kurtis.index:
    train_kurtis.loc[i,'filename'] = (str(1000000+train_kurtis.loc[i,'id'])+'.jpg')[1:]

train_women_tshirt['filename'] = train_women_tshirt['id'].astype(str).copy(deep=True)
for i in train_women_tshirt.index:
    train_women_tshirt.loc[i,'filename'] = (str(1000000+train_women_tshirt.loc[i,'id'])+'.jpg')[1:]

train_women_top_tunics['filename'] = train_women_top_tunics['id'].astype(str).copy(deep=True)
for i in train_women_top_tunics.index:
    train_women_top_tunics.loc[i,'filename'] = (str(1000000+train_women_top_tunics.loc[i,'id'])+'.jpg')[1:]


store_list = [train_men_tshirt, train_sarees,train_kurtis,train_women_tshirt,train_women_top_tunics]
store_list_names = Attribute_list['Category'].unique()
# store_list_names =['train_men_tshirt', 'train_sarees,train_kurtis','train_women_tshirt','train_women_top_tunics']



attributes_list = []
for dataset in store_list:
    attributes_list.append(list(dataset.columns[3:-1].values))




# Custom BuildingClassifierWithDropout definition
class BuildingClassifierWithDropout(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.5, unfreeze_layers=1, resnet_version=50):
        super(BuildingClassifierWithDropout, self).__init__()
        
        # Select ResNet version and load pre-trained weights
        if resnet_version == 101:
            self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        elif resnet_version == 18:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_version == 34:
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet_version == 50:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers based on `unfreeze_layers` argument
        if unfreeze_layers >= 1:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 2:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 3:
            for param in self.model.layer3.parameters():
                param.requires_grad = True

        # Adding dropout before final FC layer (if desired)
        self.dropout_fc = nn.Dropout(dropout_prob)

        # Modify the final fully connected layer to match `num_classes`
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Pass through modified ResNet
        x = self.model(x)
        x = self.dropout_fc(x)
        return x

def extract_features(model, loader): # Take the image_loader as input, extract 2048 features as output of the conv layer.
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Extracting features'):
                images = images.to(device)
                # Forward pass through ResNet
                features = model(images)
                features = features.view(features.size(0), -1)  # Flatten
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        return np.concatenate(features_list), np.concatenate(labels_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BuildingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
    



# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize and load model
model = BuildingClassifierWithDropout(num_classes=4, resnet_version=50).to(device)


# Extract features up to the last convolutional layer
# Use nn.Sequential to keep layers until 'layer4' (before FC)
feature_extractor = nn.Sequential(*(list(model.model.children())[:-1])).to(device)
feature_extractor.eval()

# Example usage with a sample input tensor
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Batch size 1, 3 channels, 224x224 image
with torch.no_grad():
    features = feature_extractor(input_tensor)

print("Shape of convolutional features:", features.shape)


le_dict = {}
# Define the custom PyTorch Dataset class
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get filename and label
        img_name = self.dataframe.iloc[idx]['filename']
        label = self.dataframe.iloc[idx]['label']
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Updated workflow 
for dataset_train, dataset_name, attr_names_for_this_dataset in zip(store_list, store_list_names, attributes_list):

    t2 = time.time()
    
    # Filter dataset based on available images
    valid_files = []
    for filename in tqdm(dataset_train['filename'], desc='Checking files'):
        if os.path.exists(os.path.join(path, filename)):
            valid_files.append(filename)
    
    dataset_train = dataset_train[dataset_train['filename'].isin(valid_files)]
    print(f"Filtered dataset: {len(valid_files)} images found.")
    print(f"Loaded Images in {time.time() - t2} seconds.")

    # Loop through attributes
    for attr_name in tqdm(attr_names_for_this_dataset, desc=f'{dataset_name}'):
        # Separate rows with missing values
        nan_rows = dataset_train[dataset_train[attr_name].isna()]
        print(len(nan_rows))
        dataset_train = dataset_train.dropna(subset=[attr_name])

        # Encode labels
        le_object = LabelEncoder()
        dataset_train['label'] = le_object.fit_transform(dataset_train[attr_name])
        identifier = f"{dataset_name}_{attr_name}"
        le_dict[identifier] = le_object
        class_names = le_object.classes_

        # Perform train-test split
        train_subset, val_subset = train_test_split(
            dataset_train,
            test_size=test_size,
            stratify=dataset_train['label'],
            random_state=42
        )

        # Prepare data
        train_dataset = ImageDataset(train_subset, path, transform)
        val_dataset = ImageDataset(val_subset, path, transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Model definition
        model = BuildingClassifierWithDropout(
            num_classes=len(class_names), resnet_version=50, 
            unfreeze_layers=unfreeze_layers, dropout_prob=dropout_prob
        ).to(device)

        # Loss, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Train the initial model
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = 100.0 * correct / total
            scheduler.step()

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100.0 * correct / total
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)*100
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}%")
        
        # Pseudo-labeling
        if psuedo_labeling == True:
            model.eval()
            pseudo_labels = []
            pseudo_rows = []
            with torch.no_grad():
                for i, row in nan_rows.iterrows():
                    image_path = f"{path}/{row['filename']}"
                    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
                    outputs = model(image)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_label = probabilities.max(dim=1)
                    if confidence.item() > confidence_threshold:
                        pseudo_rows.append(row)
                        pseudo_labels.append(predicted_label.item())
            print(len(pseudo_labels),'pseudo_labels')
            # Combine datasets
            if pseudo_rows:
                pseudo_data = pd.DataFrame(pseudo_rows)
                pseudo_data['label'] = pseudo_labels
                dataset_combined = pd.concat([dataset_train, pseudo_data], ignore_index=True)
            else:
                dataset_combined = dataset_train

            # Reinitialize the model
            model = BuildingClassifierWithDropout(
                num_classes=len(class_names), resnet_version=50, 
                unfreeze_layers=unfreeze_layers, dropout_prob=dropout_prob
            ).to(device)

            # Prepare combined dataset
            train_dataset_combined = ImageDataset(dataset_combined, path, transform)
            train_loader_combined = DataLoader(dataset=train_dataset_combined, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            # Train on combined data
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            print('-------------pseudo_label_start------------')
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                model.train()
                for inputs, labels in train_loader_combined:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                train_loss = running_loss / len(train_loader.dataset)
                train_accuracy = 100.0 * correct / total
                scheduler.step()
            
                # Validation loop
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = 100.0 * correct / total
                val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)*100
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}%")
        else:
            pass
        # Compute confusion matrix for the fold
        cm = confusion_matrix(val_labels, val_preds, labels=range(len(class_names)))
        # stores_cm.append(cm)

        # Compute and plot mean confusion matrix
        comments = f'{dataset_name}_{attr_name}'

        mean_conf_matrix = cm
        mean_conf_matrix_normalized = mean_conf_matrix.astype('float') / mean_conf_matrix.sum(axis=1)[:, np.newaxis]
        torch.save(model.state_dict(), f'{model_save_path}/Pytorch_{comments}_ep{num_epochs}_batch{batch_size}.pth')
        plt.figure(figsize=(8, 6))
        sns.heatmap(mean_conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Mean Normalized Confusion Matrix {identifier}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{confusion_save_path}/CM_{comments}_1.jpg')
        plt.show()


# Save the label encoders and models
joblib.dump(le_dict, 'le_dict.pkl')
