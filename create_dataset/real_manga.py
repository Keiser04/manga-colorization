import os
import argparse
import shutil

def clone_image(image_path, output_folder, mults):
    # Dependiendo del número de mults, actuar de manera diferente
    if mults == 1:
        # Crear una imagen con el mismo nombre y otra con 0_dfm
        new_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_0_dfm' + os.path.splitext(os.path.basename(image_path))[1])
        if not os.path.exists(new_path):
            shutil.copy(image_path, new_path)
    elif mults > 1:
        # Crear imágenes con el mismo nombre, una con el mismo nombre_0 que irá aumentando
        # dependiendo del mult y otra con _0_dfm que hará lo mismo irá aumentando
        for i in range(mults):
            new_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + f'_{i}' + os.path.splitext(os.path.basename(image_path))[1])
            if not os.path.exists(new_path):
                shutil.copy(image_path, new_path)
            new_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + f'_{i}_dfm' + os.path.splitext(os.path.basename(image_path))[1])
            if not os.path.exists(new_path):
                shutil.copy(image_path, new_path)

# Crear un analizador de argumentos
parser = argparse.ArgumentParser(description='Clonar imágenes.')
parser.add_argument('-p', '--path', type=str, help='La ruta de la carpeta de entrada.')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

# Ruta de la carpeta de entrada
input_folder = args.path if args.path else '.'

# Número de mults
mults = int(input("Por favor, introduce el número de mults: "))

# Recorrer todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de que es una imagen
        clone_image(os.path.join(input_folder, filename), input_folder, mults)
