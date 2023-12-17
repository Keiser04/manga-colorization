import os
from PIL import Image
from termcolor import colored

# Ruta de la carpeta de entrada
input_folder = 'train/color'

# Recorrer todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de que es una imagen
        try:
            # Cargar la imagen
            img = Image.open(os.path.join(input_folder, filename))
            
            # Convertir la imagen a RGB
            rgb_img = img.convert('RGB')
            
            # Guardar la imagen RGB con el mismo nombre
            rgb_img.save(os.path.join(input_folder, filename))
            
            # Imprimir el nombre del archivo que se acaba de convertir
            print(f'La imagen {filename} ha sido convertida a RGB.')
        except OSError:
            print(colored(f'La imagen {filename} está truncada o incompleta y no se pudo convertir.', 'red'))
