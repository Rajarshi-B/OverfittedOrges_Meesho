import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import joblib
import cv2
import pandas as pd
import joblib
import pandas as pd
import joblib



image_folder = '../train_images'
train_csv_path = '../CSVs/train.csv'
output_path = '../CSVs/train_filled_30_test.csv'


def perceptual_hash(image_path):
    """Compute a perceptual hash of an image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (16, 16))  # Resize for hashing
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    avg = img.mean()  # Calculate average
    img_hash = (img > avg).astype(int)  # Create binary hash
    # return img_hash.flatten()
    return ''.join(str(i) for i in img_hash.flatten())

def calculate_hash(file_path):
    """Calculate the perceptual hash of an image."""
    # Replace `perceptual_hash` with the actual hash function you're using
    return perceptual_hash(file_path), file_path

def find_duplicates(image_folder):
    """Find duplicate images in a given folder """
    hashes = {}
    duplicates = []
    unique_hashes = []
    all_images_with_hashes = {}
    image_hash = {}

    # Get all image files in the folder
    image_files = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Parallel computation of perceptual hashes with tqdm
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(calculate_hash, file_path): file_path for file_path in image_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(image_files), desc="Processing Images"):
            img_hash, file_path = future.result()
            filename = os.path.basename(file_path)
            img_hash_tuple = tuple(img_hash)
            image_hash[filename] = img_hash
            all_images_with_hashes.setdefault(img_hash_tuple, []).append(filename)

            if img_hash_tuple in hashes:
                duplicates.append((file_path, hashes[img_hash_tuple]))
            else:
                hashes[img_hash_tuple] = file_path
                unique_hashes.append(hashes)

    return duplicates, unique_hashes, all_images_with_hashes, image_hash


duplicates,unique_hashes,all_images_with_hashes,image_corresponding_hash = find_duplicates(image_folder)





train_csv = pd.read_csv(train_csv_path)
# train_csv.head()


################# This part is slow #########
for i in tqdm(train_csv.index):
    train_csv.loc[i,'filename'] = (str(1000000+train_csv.loc[i,'id'])+'.jpg')[1:]
# train_csv.head()


# tqdm.pandas()  # To enable tqdm with pandas
# Using vectorized string operations
# train_csv['filename'] = train_csv['id'].apply(lambda x: f"{x + 000000:07}.jpg")

image_hash_dict = image_corresponding_hash
len(image_hash_dict)

train_csv['image_hash'] = train_csv['filename'].copy(deep = True)


train_csv['image_hash'] = train_csv['filename'].map(image_hash_dict)

# Making a column that concatenates Category with hash

train_csv_with_hash = train_csv.copy(deep=True)

train_csv_with_hash['Category_Hash'] = train_csv_with_hash['Category'] + train_csv_with_hash['image_hash']



train_csv_with_hash

data = train_csv_with_hash.copy(deep=True)

# Group by 'image_hash' and filter to find those with multiple categories
duplicate_hashes = (
    data.groupby('Category_Hash')
    .filter(lambda x: x['filename'].nunique() > 1)
)
len(duplicate_hashes)
result = duplicate_hashes.copy(deep=True)



# Keep ids of files that match with hashes of the same category, mode is save in the same dict
# Use this dictionary to remove files
# For these ids, replace them with their modes(saved in 'mode' key), keep 1 remove rest
# 5000/30000 will be kept
store_keep_1_replace_with_mode = {'id':[],'mode':[],'filenames':[],'Category_Hash':[]}
for i in tqdm(range(result['Category_Hash'].nunique())):
    ids = result[result['Category_Hash']== result['Category_Hash'].iloc[i]].id.values
    filenames = result[result['Category_Hash']== result['Category_Hash'].iloc[i]].filename.values

    store_keep_1_replace_with_mode['id'].append(ids)

    mode = result[result['Category_Hash']== result['Category_Hash'].iloc[i]][[f'attr_{j}' for j in range(1,11)]].mode()
    store_keep_1_replace_with_mode['mode'].append(mode)
    store_keep_1_replace_with_mode['filenames'].append(filenames)
    store_keep_1_replace_with_mode['Category_Hash'].append(result['Category_Hash'].iloc[i])




keep = store_keep_1_replace_with_mode

# Load the CSV file
data = pd.read_csv(train_csv_path)

# Iterate over each entry in the keep['id'] lists
for keep_ids, mode_df in tqdm(zip(keep['id'], keep['mode'])):
    # Check if the 0th ID in the current keep_ids list is in the CSV
    primary_id = keep_ids[0]
    if primary_id in data['id'].values:
        # Get the index of the row with this primary_id
        index = data.index[data['id'] == primary_id].tolist()[0]
        
        # Fill all values in the row with values from the 0th mode
        for col in mode_df.columns:
            if col in data.columns:  # Ensure the column exists in the CSV
                data.at[index, col] = mode_df[col].iloc[0]
        
        # Collect other IDs from keep_ids (excluding the primary id) to delete them
        ids_to_delete = [other_id for other_id in keep_ids if other_id != primary_id]
        
        # Delete rows with the other IDs directly from the data
        data = data[~data['id'].isin(ids_to_delete)]
    else:
        # If the primary_id is not in the CSV, skip to the next keep_ids list
        print(f"Skipping primary ID {primary_id} as it is not in the CSV.")

# Save the updated CSV
data.to_csv(output_path, index=False)
print(f"Filled values for 0th IDs in matching rows and deleted other IDs. Saved to {output_path}.")