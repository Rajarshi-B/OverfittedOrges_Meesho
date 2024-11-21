import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.metrics import confusion_matrix,f1_score
from torchvision.models import resnet18,resnet50, ResNet18_Weights,ResNet34_Weights, VGG16_Weights, resnet50, ResNet50_Weights, ResNet101_Weights
import joblib
from imblearn.over_sampling import ADASYN
import os

################ ALWAYS CHANGE BEFORE RUNNIN #############
tag = 'duplicate_removed_cropped' # THIS SHOULD BE SAME AS THAT USED IN PREPROCESSING(pickle_dump)
##########################################################
Category_list = ['Men Tshirts','Sarees','Kurtis','Women Tshirts','Women Tops & Tunics'] # CHANGE CATEGORY
path_to_pkls = '/home/ravindra/meesho/pickels_data_FOR_TRAINING_cropped'

num_epochs = 20
learning_rate = 1e-3
optimizer_type = 'adam' # Or SGD
momentum = 0.9 # For SGD
weight_decay = 1e-5 # For SGD
# dropout_prob = 0.5
sch_step_size = 9
sch_gamma = 0.1
scheduler_on = True
batch_size_list = [64]
unfreeze_list = [2]  # Number of Layers to unfreeze
dropout_prob_list = [0.25]
resnet_version_list = [50]            

oversample = False


folder_tag = f'{tag}_KFOLD_Run_RESNET{resnet_version_list}_OverSampling{oversample}_unfreeze{unfreeze_list}' # ALWAYS CHANGE THIS BEFORE RUNNING

Top_directory = f'{folder_tag}_RESNET_MODELS'

model_folder = 'Models'
accuracy_folder = 'Accuracy_curve'
confusion_folder = 'Confusion_matrix'

try:
    os.mkdir(f'/home/ravindra/meesho/MODELS/{Top_directory}')
except:
    print('CHANGE THE FOLDER TAG AND RUN AGAIN!')
    exit()

model_save_path = f'/home/ravindra/meesho/MODELS/{Top_directory}/{model_folder}'
accuracy_save_path = f'/home/ravindra/meesho/MODELS/{Top_directory}/{accuracy_folder}'
confusion_save_path = f'/home/ravindra/meesho/MODELS/{Top_directory}/{confusion_folder}'


try:
    os.mkdir(model_save_path)
    os.mkdir(accuracy_save_path)
    os.mkdir(confusion_save_path)
except:
    print('CANNOT MAKE SUBFOLDERS!')
    exit()


for i in Category_list:
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

    for j in [f'attr_{k}' for k in range(1,z+1)]:
        attr_name = j
        remar = i
        image_array = joblib.load(f'{path_to_pkls}/image_array_{remar}_{attr_name}_{tag}.pkl')
        image_labels = joblib.load(f'{path_to_pkls}/labels_{remar}_{attr_name}_{tag}.pkl')

        class_names = joblib.load(f'{path_to_pkls}/Class_names_list_{attr_name}_{remar}_{tag}.pkl')


        target = image_labels.astype(int)

        print(f'category {i}, attribute {j},no of {z}')
        # Hyperparameters
        num_classes = np.max(image_labels)+1
        
        
        comments = f'{remar}_{attr_name}'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset class for loading data
        class BuildingDataset(Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx].astype(np.uint8)  # Ensure image data is in uint8
                image = Image.fromarray(image)  # Convert NumPy array to PIL image
                
                if self.transform:
                    image = self.transform(image)
                
                return image, self.labels[idx]

        # Training function
        def train_model():
            train_acc, val_acc = [], []
            
            for epoch in range(num_epochs):
                model.train()
                correct_train = 0
                total_train = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Calculate training accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                
                train_acc.append(100 * correct_train / total_train)

                # Validation phase
                model.eval()
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                val_acc.append(100 * correct_val / total_val)

                # Update learning rate if scheduler is used
                if scheduler_on:
                    scheduler.step()
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Train Accuracy: {train_acc[-1]:.2f}%, Validation Accuracy: {val_acc[-1]:.2f}%, LR: {scheduler.get_lr()}')
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Train Accuracy: {train_acc[-1]:.2f}%, Validation Accuracy: {val_acc[-1]:.2f}%')

                # Save model checkpoint periodically
                if epoch % 20 == 0:
                    val_accuracy = 100 * correct_val / total_val
                    # torch.save(model, f'/home/ravindra/meesho/_vikas_/check_point_models/Pytorch_{remarks}_val{val_accuracy}%_ep{epoch}_checkpoint.pth')
                    # print(f'Checkpoint Saved: dropout = {dropout_prob}, unfreeze layers = {layers_unfreeze}')

            return train_acc, val_acc

        # Main training loop with different configurations
        for resnet_version in resnet_version_list:
            remarks = f'{comments}_Train_ResNET{resnet_version}_scheduler{scheduler_on}_SMOTE{oversample}_optim{optimizer_type}'

            class BuildingClassifierWithDropout(nn.Module):
                def __init__(self, num_classes=5, dropout_prob=0.5, unfreeze_layers=1):
                    super(BuildingClassifierWithDropout, self).__init__()
                    # Using pretrained ResNet101 for transfer learning
                    if(resnet_version == 101):
                        self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
                    elif(resnet_version == 18):
                        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                    elif(resnet_version == 34):
                        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
                    elif(resnet_version == 50):
                        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

                    # Freeze all layers first
                    for param in self.model.parameters():
                        param.requires_grad = False

                    # Option to unfreeze more layers
                    if unfreeze_layers == 1:
                        for param in self.model.fc.parameters():
                            param.requires_grad = True
                    elif unfreeze_layers == 2:
                        for param in self.model.layer4.parameters():
                            param.requires_grad = True
                        for param in self.model.fc.parameters():
                            param.requires_grad = True
                    elif unfreeze_layers == 3:
                        for param in self.model.layer3.parameters():
                            param.requires_grad = True
                        for param in self.model.layer4.parameters():
                            param.requires_grad = True
                        for param in self.model.fc.parameters():
                            param.requires_grad = True
                    else:
                        for param in self.model.parameters():
                            param.requires_grad = True

                    
                    self.dropout_fc = nn.Dropout(dropout_prob)    # Dropout before the final fully connected layer
                    
                    # Modify the final fully connected layer
                    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                
                
                def forward(self, x):
                    # Pass through the ResNet18 feature extractor
                    x = self.model(x)
                    # Apply dropout before the final layer
                    x = self.dropout_fc(x)
                    return x
            transform = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
                transforms.Resize((224, 224)), # Add random rotation + horizontal flipping
                # transforms.RandomHorizontalFlip(p=0.5),       # Add random horizontal flip
                # transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])
            
            

            for batch_size in batch_size_list:
                X_train, X_val, y_train, y_val = train_test_split(
                    image_array, target, train_size=0.8, stratify=target, random_state=42
                )
                
                if oversample:
                    # Apply ADASYN for handling class imbalance
                    print(X_train.shape)
                    X = X_train.reshape((X_train.shape[0], -1))
                    try:
                        adasyn = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
                        X_resampled, y_resampled = adasyn.fit_resample(X, y_train)
                    except:
                        try:
                            adasyn = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=4)
                            X_resampled, y_resampled = adasyn.fit_resample(X, y_train)
                        except:
                            adasyn = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
                            X_resampled, y_resampled = adasyn.fit_resample(X, y_train)
                    
                    # Reshape the resampled data back to the original image dimensions
                    X_resampled = X_resampled.reshape(
                        (X_resampled.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
                    )
                    
                    train_dataset = BuildingDataset(X_resampled, y_resampled, transform=transform)
                    val_dataset = BuildingDataset(X_val, y_val, transform=transform)
                else:
                    train_dataset = BuildingDataset(X_train, y_train, transform=transform)
                    val_dataset = BuildingDataset(X_val, y_val, transform=transform)


                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                for dropout_prob in dropout_prob_list:
                    for layers_unfreeze in unfreeze_list:
                        # Model initialization
                        model = BuildingClassifierWithDropout(num_classes=num_classes, dropout_prob=dropout_prob, unfreeze_layers=layers_unfreeze).to(device)

                        # Loss and optimizer
                        criterion = nn.CrossEntropyLoss()
                        if optimizer_type == 'sgd':
                            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                        else:
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        
                        # Learning rate scheduler
                        if scheduler_on:
                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sch_step_size, gamma=sch_gamma)

                        # Train the model
                        train_acc, val_acc = train_model()

                        # Save the trained model
                        # try:
                            # os.mkdir('{remarks}_ep{num_epochs}_batch{batch_size}')
                        torch.save(model.state_dict(), f'{model_save_path}/Pytorch_{remarks}_ep{num_epochs}_batch{batch_size}.pth')
                        
                        # Plot accuracy curves
                        plt.figure(figsize=(10, 5))
                        plt.plot(range(num_epochs), train_acc, label="Train Accuracy")
                        plt.plot(range(num_epochs), val_acc, label="Validation Accuracy")
                        plt.xlabel('Epoch')
                        plt.ylabel('Accuracy (%)')
                        plt.title('Train vs Validation Accuracy')
                        plt.legend()
                        plt.savefig(f'{accuracy_save_path}/{remarks}_batch{batch_size}.jpg')
                        plt.close()

                        # Confusion Matrix calculation and plotting
                        def compute_confusion_matrix_and_f1(model, val_loader):
                            
                            all_preds, all_labels = [], []
                            model.eval()
                            with torch.no_grad():
                                for images, labels in val_loader:
                                    images, labels = images.to(device), labels.to(device)
                                    outputs = model(images)
                                    _, preds = torch.max(outputs, 1)
                                    all_preds.extend(preds.cpu().numpy())
                                    all_labels.extend(labels.cpu().numpy())
                            
                            # Compute confusion matrix
                            cm = confusion_matrix(all_labels, all_preds)
                            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                            
                            # Compute F1 score
                            f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for multi-class
                            
                            return cm_normalized, f1

                        # Visualization and saving the confusion matrix
                        plt.figure(figsize=(12, 12))
                        cm_normalized, f1_score_value = compute_confusion_matrix_and_f1(model, val_loader)

                        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                                    yticklabels=class_names)

                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Confusion Matrix {comments}_{remarks}\nF1 Score: {f1_score_value:.2f}')
                        plt.savefig(f'{confusion_save_path}/CM_{comments}_{remarks}_1.jpg')
                        plt.close()

                        # Clean up and empty GPU memory
                        del model
                        torch.cuda.empty_cache()
