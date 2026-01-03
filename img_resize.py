import cv2
import os

input_dir = "../training/raw_good_yam"  # Folder contains images directly
output_dir = "../training/Yam/good_yam"
target_size = (256, 256)

os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f" Skipping unreadable file: {image_path}")
        continue

    # Resize the image
    resized_img = cv2.resize(img, target_size)

    # Save to output folder
    save_path = os.path.join(output_dir, image_name)
    cv2.imwrite(save_path, resized_img)
    print(f" Saved: {save_path}")

print(" All images resized and saved to 'resized_dataset/'")
