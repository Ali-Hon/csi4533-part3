import os
from PIL import Image
import torch
from utils import model, tools

# Load the segmentation model and its associated transforms
seg_model, get_transform = model.get_model()

# Define paths
source_path_dir = "reference_people/input"
output_path_dir = "reference_people/output"

def generate_ref_masks():
    for image_name in os.listdir(source_path_dir):
        # Load the image
        image_path = os.path.join(source_path_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Apply model-specific transformations
        transformed_img = get_transform(image)
        input_tensor = transformed_img.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            prediction = seg_model(input_tensor)

        # Process the prediction and save the result
        result = tools.process_inference(prediction, image, os.path.join('reference_people', 'saved_masks', image_name.removesuffix('.png') + '.npy'))
        result.save(os.path.join(output_path_dir, "masked_" + image_name))