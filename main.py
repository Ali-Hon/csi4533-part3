import os
from PIL import Image
from utils import model, tools
import torch

# Point d'entrée principal du script
if __name__ == "__main__":

    # Définir les répertoires source et de sortie, et le nom de l'image
    source_path_dir = "examples/source"
    output_path_dir = "examples/output"
    image_name = "sample_1.png"

    # Charger le modèle et appliquer les transformations à l'image
    seg_model, transforms = model.get_model()

    # Ouvrir l'image et appliquer les transformations
    image_path = os.path.join(source_path_dir, image_name)
    image = Image.open(image_path)
    transformed_img = transforms(image)
    

    # Effectuer l'inférence sur l'image transformée sans calculer les gradients
    with torch.no_grad():
        output = seg_model([transformed_img])

    # Traiter le résultat de l'inférence
    result = tools.process_inference(output,image)
        
    ## (optional) apply saved mask
    #result = tools.apply_saved_mask(image)
    
    result.save(os.path.join(output_path_dir, image_name))
    # result.show()