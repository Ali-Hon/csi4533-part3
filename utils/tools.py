import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_box(mask, img_path, target_path, img_name):
    """
    Function to draw box for on the image on which a person was potentially detected using its mask
    """
    # Logic: Find bbox using binary mask
    bbox = get_boundingbox(mask)
    # open the image
    image = Image.open(img_path)
    # draw the bbox on the image
    draw_bb([bbox], image)
    # IMPORTANT: save it in the appropriate person output folder as wanted for the results of the part3
    image.save(os.path.join(target_path, img_name))

# get bbox boundaries where numpy_mask > 0, i.e. where the pixel is a 1
# this means that the person is there on the image, so we can draw a mask around that
def get_boundingbox(numpy_mask):
   y_indices, x_indices = np.where(numpy_mask > 0)

   x_min = np.min(x_indices)
   x_max = np.max(x_indices)
   y_min = np.min(y_indices)
   y_max = np.max(y_indices)

   bounding_box = [(x_min, y_min), (x_max, y_max)]

   return bounding_box

def draw_bb(boxes, image):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline='green', width=3)

# Fonction pour traiter les sorties d'inférence du modèle
def process_inference(model_output, image, path, box_mask=None):
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
                
    # save masks for the image at the appropriate path
    masks_array = np.array(np_masks, dtype=bool)
    with open(path, 'wb') as f:
        np.save(f, masks_array)
        print(f"Saved masks to {path}")

    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))

# read saved masks from npy file and return them their 2D array
def read_saved_masks(path):
    saved_masks = np.load(path)
    return saved_masks

# fonction que nous avons utilisé pour dessiner un histogramme et voir
# s'il avait de l'allure -- résultat était bon mais nous n'avons pas
# eu besoin de l'utiliser par la suite après avoir vérifier
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