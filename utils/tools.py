import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour traiter les sorties d'inférence du modèle
def process_inference(model_output, image, path):
    np_masks = []
    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for _, (mask, score, label) in enumerate(zip(masks, scores, labels)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255) 
            np_masks.append(mask_np)            

            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
                
    masks_array = np.array(np_masks, dtype=bool)
    with open(path, 'wb') as f:
        np.save(f, masks_array)
        print(f"Saved masks to {path}")

    ## (Optional) save mask
    # with open('examples/output/saved_masks.npy', 'wb') as f:
    #     np.save(f,np_masks)

    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))

def apply_saved_mask(image):
    # Convertir l'image en tableau numpy
    img_np = np.array(image)
    masks = np.load('examples/output/saved_masks.npy')
    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for _, mask in enumerate(masks):  
        for c in range(3):
            img_np[:, :, c] = np.where(mask, 
                                    (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                    img_np[:, :, c])
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer
    return Image.fromarray(img_np.astype(np.uint8))

# def read_saved_mask():
#     masks = np.load('images_output/cam0/saved_masks/1637433774251426200_saved_masks.npy')
#     for mask in masks:
#         print(mask.astype(np.uint8))
        
def read_saved_masks(path):
    saved_masks = np.load(path)
    return saved_masks
    # grayscale_np = np.array(grayscale_image)

    # for idx, person_mask in enumerate(saved_masks):
    #     segmented_person = grayscale_np * person_mask
        
    #     plot_grayscale_histogram(segmented_person, idx + 1)



    # saved_masks = np.load(path)
    # print(saved_masks)
    # # Assuming the first mask corresponds to "person 1" and is of interest
    # person_1_mask = saved_masks[0]

    # # Convert the original image to grayscale
    # grayscale_image = Image.open(image_path).convert('L')
    # grayscale_np = np.array(grayscale_image)

    # # Apply the mask to the grayscale image to segment "person 1"
    # segmented_person = grayscale_np * person_1_mask

    # # Calculate and plot the grayscale histogram for the segmented area
    # print(segmented_person.astype(np.uint8))
    # # plot_grayscale_histogram(segmented_person)
    # return person_1_mask.astype(np.uint8)

def plot_grayscale_histogram(segmented_person):
    """
    Function to calculate and plot the histogram for a grayscale image.
    """
    histogram, bin_edges = np.histogram(
        segmented_person[segmented_person > 0], bins=256, range=(0, 255)
    )
    plt.figure(figsize=(8, 6))
    plt.plot(bin_edges[0:-1], histogram)  # Plot the histogram
    plt.title('Grayscale Histogram for Segmented Person')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    plt.show()