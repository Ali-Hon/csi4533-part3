import os
import cv2 as cv
from PIL import Image
from utils import model, tools
import torch
import generate_reference
import time
import numpy as np

def calculate_histogram(mask_path, bins=256):
    mask = np.load(mask_path)
    hist, _ = np.histogram(mask, bins=bins, range=(0, 256))
    hist = hist.astype('float32') / hist.sum()
    return hist

# Point d'entrée principal du script
if __name__ == "__main__":
    # Définir les répertoires source et de sortie, et le nom de l'image
    source_path_dir = "images/"
    output_path_dir = "images_output/"

    skip = True
    # get masks for reference people
    if not skip:
        generate_reference.generate_ref_masks()

    # make histograms from reference people masks
    reference_histograms = []
    path_to_masks = os.path.join('reference_people', 'saved_masks')
    for person in os.listdir(path_to_masks):
        hist = calculate_histogram(os.path.join(path_to_masks, person))
        reference_histograms.append(hist)
    
    # get half histograms
    half_histograms = []
    
    # generating images with masks as per part2
    # also generating npy files for each image with the masks saved
    # so we can read them later and do computations using them
    if not skip:
        for feed in ['cam0', 'cam1']:
            for image_name in os.listdir(os.path.join(source_path_dir, feed)):
                if image_name in os.listdir(os.path.join(output_path_dir, feed)):
                    continue

                # Charger le modèle et appliquer les transformations à l'image
                seg_model, transforms = model.get_model()

                # Ouvrir l'image et appliquer les transformations
                image_path = os.path.join(source_path_dir, feed, image_name)
                image = Image.open(image_path)
                transformed_img = transforms(image)

                # Effectuer l'inférence sur l'image transformée sans calculer les gradients
                with torch.no_grad():
                    output = seg_model([transformed_img])

                # current path where we save the npy file of masks of the current image being generated
                current_saved_masks_path = os.path.join(output_path_dir, feed, 'saved_masks', image_name.removesuffix('.png') + '_saved_masks.npy')
                # Traiter le résultat de l'inférence
                result = tools.process_inference(output, image, current_saved_masks_path)
                    
                ## (optional) apply saved mask
                # result = tools.apply_saved_mask(image)
                
                result.save(os.path.join(output_path_dir, feed, image_name))
                # result.show()

    results = []
    corresponding_filename = []
    # compare each reference person's histogram to the histogram the 'masked' people from part2's RCNN
    for i, person_histogram in enumerate(reference_histograms):
        for feed in ['cam0', 'cam1']:
            for mask_filename in os.listdir(os.path.join(output_path_dir, feed, 'saved_masks')):
                print(os.path.join(output_path_dir, feed, 'saved_masks', mask_filename))
                current_masks = tools.read_saved_masks(os.path.join(output_path_dir, feed, 'saved_masks', mask_filename))
                for j, mask in enumerate(current_masks):
                    print(mask_filename)
        
                    hist = calculate_histogram(os.path.join('reference_people', 'saved_masks', person))
                    
                    correlation = cv.compareHist(person_histogram, hist, cv.HISTCMP_CORREL)
                    print(f"Person {i} Histogram Comparison with Mask {j} in file {mask_filename}: {correlation}")

                    if len(results) < 100:
                        results.append(correlation)
                        corresponding_filename.append(mask_filename)
                    elif min(results) < correlation:
                        i = results.index(min(results))
                        results.remove(min(results))
                        corresponding_filename.remove(corresponding_filename[i])

                        results.append(correlation)
                        corresponding_filename.append(mask_filename)

    d = zip(results, corresponding_filename)
    for el in d:
        print(el)