import os
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import clip
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


train_csv=pd.read_csv('../CSVs/train_filled_30.csv')
# Set up paths and parameters
image_folder = '../train_images'
output_folder = '../representative_images'  # Folder to save selected images
n_clusters = 200  # Adjust based on desired diversity in selected images


Attribute_name_actual = pd.read_parquet('..CSVs/category_attributes.parquet')

for cat in Attribute_name_actual['Category'].values:
    train_csv = train_csv[train_csv['Category'] == cat].copy(deep=True)

    description_text = []
    for i in tqdm(range(len(train_csv))):
        id = train_csv['id'].iloc[i]
        attr_name_list = Attribute_name_actual[Attribute_name_actual['Category'] == train_csv['Category'].iloc[i]]['Attribute_list']
        s = ''
        # print(attr_name_list)
        for j,attr_name in enumerate(attr_name_list.values[0]):
            attr_val = train_csv[f'attr_{j+1}'].iloc[i]
            if(attr_val is not np.nan):
                s = s + f'{attr_name} is {attr_val}, '
            else:
                pass
        description_text.append(s)

    train_csv['Description'] = description_text


    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Function to extract features using CLIP
    def extract_clip_features(image_path):
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return features.cpu().numpy().flatten()

    # Load and process all images
    feature_list = []
    image_paths = []

    def combine_features(image_features, text_features, method="concatenate"):
        if method == "concatenate":
            combined_features = np.concatenate((image_features, text_features))
        elif method == "average":
            combined_features = (image_features + text_features) / 2
        elif method == "weighted_sum":
            # Example with weights for image and text contributions
            alpha, beta = 0.6, 0.4
            combined_features = alpha * image_features + beta * text_features
        else:
            raise ValueError("Unknown combination method")
        return combined_features

    def extract_clip_image_features(image_path):
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features.cpu().numpy().flatten()

    # Function to extract features for text
    def extract_clip_text_features(text):
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        return text_features.cpu().numpy().flatten()


    for i in tqdm(range(len(train_csv))):
        filename = str(train_csv['id'].iloc[i]).zfill(6) + '.jpg'
        if(filename in os.listdir(image_folder)):
            image_path = os.path.join(image_folder, filename)
            features = extract_clip_features(image_path)
            image_features = extract_clip_image_features(image_path)
            text_features = extract_clip_text_features(train_csv['Description'].iloc[i])
            features = combine_features(image_features, text_features, method="weighted_sum")
            feature_list.append(features)
            image_paths.append(image_path)

    # Convert features to a numpy array
    feature_array = np.array(feature_list)

    # print()
    # Clustering using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(feature_array)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Select the closest image to each cluster center
    selected_images = []
    for i in tqdm(range(n_clusters)):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = feature_array[cluster_indices]
        center = cluster_centers[i]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        selected_images.append(image_paths[closest_index])

    # Save selected images to the output folder
    os.makedirs(output_folder, exist_ok=True)
    for img in tqdm(selected_images):
        shutil.copy(img, os.path.join(output_folder, os.path.basename(img)))

    print(f"Selected images have been saved to {output_folder}")

    # Visualization with t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    feature_2d = tsne.fit_transform(feature_array)

    # Plot clusters
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        # Get all points for this cluster
        cluster_points = feature_2d[labels == i]
        # Plot each cluster with a different color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', s=10)

    # Add title and labels
    plt.title(f"Image Clusters Visualization {cat}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.show()