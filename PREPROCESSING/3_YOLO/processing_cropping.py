import os
from PIL import Image
from ultralytics import YOLO

# Load the model (fine-tuned model path)
model = YOLO("./runs/detect/train/weights/best.pt")



# for train images

images_folder = "SUBMISSION_STRUCTURED/train_Images" # path to train image folder without duplicates
output_folder = "SUBMISSION_STRUCTURED/Train_Images_CROPPED" # output path for cropped train images

os.makedirs(output_folder, exist_ok=True)

image_files = [i for i in os.listdir(images_folder) if i.endswith('.jpg')]

not_detected_images = []

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    
    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}, skipping...")
        continue  
    
    # Perform prediction
    results = model(image_path)
    
    # Get the original image
    original_image = Image.open(image_path)
    
    # Initialize variables to find the minimum enclosing box
    x_min, y_min, x_max, y_max = None, None, None, None
    
    # Loop over each result to find all bounding boxes
    for result in results:
        for box in result.boxes.xyxy:  # xyxy format [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box
            if x_min is None or x1 < x_min:
                x_min = x1
            if y_min is None or y1 < y_min:
                y_min = y1
            if x_max is None or x2 > x_max:
                x_max = x2
            if y_max is None or y2 > y_max:
                y_max = y2
    
    # Check if any valid bounding box was found
    if x_min is None or y_min is None or x_max is None or y_max is None:
        # If no bounding boxes found, copy the original image to the output folder
        output_path = os.path.join(output_folder, f"{image_file}")
        original_image.save(output_path)
        print(f"No bounding boxes found for {image_file}, image copied.")
        not_detected_images.append(image_file)
        continue
    
    # Crop the image using the minimum enclosing bounding box
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image
    output_path = os.path.join(output_folder, f"{image_file}")
    cropped_image.save(output_path)

# Print summary of images not detected and copied
print(f"\nTotal images not detected in train and copied: {len(not_detected_images)}")
if len(not_detected_images) > 0:
    print("Images not detected and copied:")
    for image_file in not_detected_images:
        print(image_file)




# for test images

images_folder = "........../test_images" # path to test image folder 
output_folder = "......./test_images_cropped" # output path for cropped test images

os.makedirs(output_folder, exist_ok=True)

image_files = [i for i in os.listdir(images_folder) if i.endswith('.jpg')]

not_detected_images = []

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    
    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}, skipping...")
        continue  
    
    # Perform prediction
    results = model(image_path)
    
    # Get the original image
    original_image = Image.open(image_path)
    
    # Initialize variables to find the minimum enclosing box
    x_min, y_min, x_max, y_max = None, None, None, None
    
    # Loop over each result to find all bounding boxes
    for result in results:
        for box in result.boxes.xyxy:  # xyxy format [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box
            if x_min is None or x1 < x_min:
                x_min = x1
            if y_min is None or y1 < y_min:
                y_min = y1
            if x_max is None or x2 > x_max:
                x_max = x2
            if y_max is None or y2 > y_max:
                y_max = y2
    
    # Check if any valid bounding box was found
    if x_min is None or y_min is None or x_max is None or y_max is None:
        # If no bounding boxes found, copy the original image to the output folder
        output_path = os.path.join(output_folder, f"{image_file}")
        original_image.save(output_path)
        print(f"No bounding boxes found for {image_file}, image copied.")
        not_detected_images.append(image_file)
        continue
    
    # Crop the image using the minimum enclosing bounding box
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image
    output_path = os.path.join(output_folder, f"{image_file}")
    cropped_image.save(output_path)

# Print summary of images not detected and copied
print(f"\nTotal images not detected in test and copied: {len(not_detected_images)}")
if len(not_detected_images) > 0:
    print("Images not detected and copied:")
    for image_file in not_detected_images:
        print(image_file)