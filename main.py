import os
import cv2 as cv
from PIL import Image
from utils import model, tools
import torch
import generate_reference
import numpy as np

def calculate_histograms(mask_path, bins=256):
    """calculates full and half histograms"""
    mask = np.load(mask_path)
    # Calculate the full mask histogram
    full_hist, _ = np.histogram(mask, bins=bins, range=(0, 256))
    full_hist = full_hist.astype('float32') / full_hist.sum()

    # Calculate the top half mask histogram
    num_rows = mask.shape[0] // 2
    top_half_mask = mask[:num_rows, :]
    top_half_hist, _ = np.histogram(top_half_mask, bins=bins, range=(0, 256))
    top_half_hist = top_half_hist.astype('float32') / top_half_hist.sum()

    return full_hist, top_half_hist

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
        full_hist, top_half_hist = calculate_histograms(os.path.join(path_to_masks, person))
        reference_histograms.append({'full': full_hist, 'top_half': top_half_hist})
    
    # generating images with masks as per part2
    # also generating npy files for each image with the masks saved
    # so we can read them later and do computations using them
    if not skip:
        for feed in ['cam0', 'cam1']:
            for image_name in os.listdir(os.path.join(source_path_dir, feed)):
                if image_name in os.listdir(os.path.join(output_path_dir, feed)):
                    continue
                
                print(image_name)

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

    # final results for all reference people stored here
    end_results = []
    end_corresponding_filename = []
    # folder where results need to be stored
    target_folder = ['person1','person2','person3','person4','person5']
    # compare each reference person's histogram to the histogram the 'masked' people from part2's RCNN
    for i, person_histogram_dict in enumerate(reference_histograms):
        # file where results are saved for top 100 comparaisons for current person
        f = open(os.path.join('output_results', target_folder[i], f'output_{target_folder[i]}.txt'), 'w')
        for ref_hist_key in ['full', 'top_half']:
            # lists for results of current lists
            # results[i] was found in the saved_masks file at corresponding_filename[i]
            results = []
            corresponding_filename = []
            for feed in ['cam0', 'cam1']:
                for mask_filename in os.listdir(os.path.join(output_path_dir, feed, 'saved_masks')):
                    # if mask_filename in os.listdir(os.path.join(output_path_dir, feed, 'saved_masks')):
                    #     continue

                    current_masks = tools.read_saved_masks(os.path.join(output_path_dir, feed, 'saved_masks', mask_filename))
                    for j, mask in enumerate(current_masks):
                        full_hist, half_hist = calculate_histograms(os.path.join('reference_people', 'saved_masks', person))
                        for dataset_hist in [full_hist, half_hist]:
                            correlation = cv.compareHist(person_histogram_dict[ref_hist_key], dataset_hist, cv.HISTCMP_CORREL)
                            out_s = f"Person {i + 1} Histogram Comparison with Mask {j} in file {mask_filename}: {correlation}"
                            print(out_s)
                            f.write(out_s + '\n')
                            
                            img_path = os.path.join(source_path_dir, feed, mask_filename.removesuffix('_saved_masks.npy') + '.png')
                            # logic to keep best 100 results for current person
                            # fill it in up to 100
                            if len(results) < 100:
                                results.append(correlation)
                                corresponding_filename.append(mask_filename)

                                tools.draw_box(mask, img_path, os.path.join('output_results', target_folder[i]), mask_filename.removesuffix('_saved_masks.npy') + '.png')
                            # when it's 100, take out the worst (min) result and replace it with the 
                            # currently computed correlation if it's better then it
                            elif min(results) < correlation:
                                index = results.index(min(results))
                                results.remove(min(results))
                                corresponding_filename.remove(corresponding_filename[index])

                                results.append(correlation)
                                corresponding_filename.append(mask_filename)

                                tools.draw_box(mask, img_path, os.path.join('output_results', target_folder[i]), mask_filename.removesuffix('_saved_masks.npy') + '.png')

        # write the results for current person to 'end' lists
        end_results.append(results)
        end_corresponding_filename.append(corresponding_filename)

        f.close()

    print(len(end_results))
    print(len(end_corresponding_filename))
    print(len(end_results[0]))
    print(len(end_corresponding_filename[0]))
    print(len(set(end_corresponding_filename[0])))