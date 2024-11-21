import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load data
data = pd.read_csv('../CSVs/train_45k_imputed_with_0.9_threshold.csv')
path_img = '../cropped_30k_images'

tag = 'Cropped_30k'

save_to = f'../SUBMISSION_STRUCTURED/MODELS&PICKLES/pickles_{tag}'




try:
    print(f'Folder Made {save_to}')
    os.mkdir(save_to)
except:
    None


def process_images(category, attr_name):
    # Filter data for the specific category
    features_data = data[data['Category'] == category].copy(deep = True)
    features_data.loc[:,'filename'] = features_data['id'].astype(str).str.zfill(6) + '.jpg'
    train_c = features_data[['id', 'filename', attr_name]].copy(deep=True)
    image_names = set(os.listdir(path_img))
    train_c.loc[:,'image_present'] = train_c['filename'].apply(lambda x: x in image_names)
    train_c.dropna(inplace=True)
    
    y = train_c[attr_name].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(y, f'{save_to}/labels_{category}_{attr_name}_{tag}.pkl')
   

    # Initialize an array for images
    image_array = np.zeros((len(train_c), 224, 224, 3), dtype=np.uint8)
    filenames = train_c['filename'].tolist()

    def load_and_resize_image(filename):
        try:
            img = Image.open(f'{path_img}/{filename}').resize((224, 224))
            return np.array(img)
        except FileNotFoundError:
            print(f"Warning: File not found - {filename}")
            return None  # Handle missing images gracefully

    with ThreadPoolExecutor() as executor:
        results = executor.map(load_and_resize_image, filenames)
        for i, img in enumerate(results):
            if img is not None:
                image_array[i] = img

    # Save processed images and label encodings
    joblib.dump(image_array, f'{save_to}/image_array_{category}_{attr_name}_{tag}.pkl')
    joblib.dump(list(le.classes_), f'{save_to}/Class_names_list_{attr_name}_{category}_{tag}.pkl')

categories = ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics']
attributes = [f'attr_{k}' for k in range(1, 11)]

for category in tqdm(categories,desc='Making Pickles'):
    for attr_name in attributes:
        process_images(category, attr_name)