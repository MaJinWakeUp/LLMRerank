import os
import shutil

# Paths
gcloud_output_dir = "gcloud_output"
source_images_dir = "/scratch/jin7/datasets/AMMeBa/images_part1/"
destination_images_dir = "images/"

# Ensure destination directory exists
os.makedirs(destination_images_dir, exist_ok=True)

# Process files in gcloud_output
for filename in os.listdir(gcloud_output_dir):
    file_path = os.path.join(gcloud_output_dir, filename)
    if os.path.isfile(file_path):
        image_name = filename.replace('.json', '.jpg')  # Assuming the file name corresponds to the image name
        source_image_path = os.path.join(source_images_dir, image_name)
        destination_image_path = os.path.join(destination_images_dir, image_name)

        # Copy the image if it exists
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied: {image_name}")
        else:
            print(f"Image not found: {image_name}")