from PIL import Image
import os

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    # Solo realiza el resize si alguna dimensión es menor que 512
    if width < size or height < size:
        if width < height:
            new_height = size
            new_width = int(new_height * width / height)
        else:
            new_width = size
            new_height = int(new_width * height / width)

        resized_image = original_image.resize((new_width, new_height))
        resized_image.save(output_image_path)

# Solicita al usuario la ruta de la carpeta
folder_path = input("Por favor, introduce la ruta de la carpeta: ")

# Tamaño deseado
size = 512

# Recorre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        input_image_path = os.path.join(folder_path, filename)
        output_image_path = os.path.join(folder_path, "resized_" + filename)
        resize_image(input_image_path, output_image_path, size)
