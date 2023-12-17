import os
from PIL import Image

def convert_to_bw(image_path, output_path, mults):
    # Cargar la imagen
    img = Image.open(image_path).convert('L')

    # Guardar la imagen en blanco y negro con el mismo nombre
    img.save(os.path.join(output_path, os.path.basename(image_path)))

    # Dependiendo del número de mults, actuar de manera diferente
    if mults == 1:
        # Crear una imagen en bw con el mismo nombre y otra con 0_dfm
        img.save(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '_0_dfm' + os.path.splitext(os.path.basename(image_path))[1]))
    elif mults > 1:
        # Crear imágenes con bw con el mismo nombre, una con el mismo nombre_0 que irá aumentando
        # dependiendo del mult y otra con _0_dfm que hará lo mismo irá aumentando
        for i in range(mults):
            img.save(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + f'_{i}' + os.path.splitext(os.path.basename(image_path))[1]))
            img.save(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + f'_{i}_dfm' + os.path.splitext(os.path.basename(image_path))[1]))

# Ruta de la carpeta de entrada y salida
input_folder = 'train/color'
output_folder = 'train/bw'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Número de mults
mults = int(input("Por favor, introduce el número de mults: "))

# Recorrer todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de que es una imagen
        convert_to_bw(os.path.join(input_folder, filename), output_folder, mults)
