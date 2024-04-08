import os
from PIL import Image
from utils import model, tools
import torch
import generate_reference

# Point d'entrée principal du script
if __name__ == "__main__":
    # Définir les répertoires source et de sortie, et le nom de l'image
    source_path_dir = "images/"
    output_path_dir = "images_output/"

    # get masks for reference people
    generate_reference.generate_ref_masks()

    # reference_histograms: [hist1, hist2, ...]
    # make histograms from reference people masks
    reference_histograms = []
    for person_mask in os.listdir(os.path.join('reference_people', 'saved_masks')):
        reference_histograms.append(np.histogram(person_mask))

    # generating images with masks as per part2
    # also generating npy files for each image with the masks saved
    # so we can read them later and do computations using them
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
    for person_histogram in reference_histograms:
        for feed in ['cam0', 'cam1']:
            for mask_filename in os.listdir(os.path.join(output_path_dir, feed, 'saved_masks')):
                print(os.path.join(output_path_dir, feed, 'saved_masks', mask_filename))
                current_masks = tools.read_saved_masks(os.path.join(output_path_dir, feed, 'saved_masks', mask_filename))
                for mask in current_masks:
                    print(mask_filename)

                    correlation = person_histogram.compare()

                    if len(results) < 100:
                        results.append(correlation)
                        corresponding_filename.append(mask_filename)
                    elif min(results) < correlation:
                        i = results.index(min(results))
                        results.remove(min(results))
                        corresponding_filename.remove(corresponding_filename[i])

                        results.append(correlation)
                        corresponding_filename.append(mask_filename)

    print(zip(results, corresponding_filename))