import os
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def convert_to_png(image_path):
    try:
        # Cargar la imagen
        img = Image.open(image_path)

        # Guardar la imagen en formato PNG en la misma carpeta donde se encontró
        png_path = os.path.splitext(image_path)[0] + '.png'
        img.save(png_path)

        # Eliminar la imagen original si no es PNG
        if not image_path.lower().endswith('.png'):
            os.remove(image_path)

        # Imprimir el nombre de la imagen
        print(f'Imagen convertida y original eliminada: {png_path}')
    except IOError:
        print(f'No se pudo abrir la imagen: {os.path.basename(image_path)}')

def main():
    # Crear un analizador de argumentos
    parser = argparse.ArgumentParser(description='Convertir imágenes a PNG y eliminar las originales.')
    parser.add_argument('-p', '--path', type=str, help='La ruta de la carpeta de entrada.')

    # Analizar los argumentos de la línea de comandos
    args = parser.parse_args()

    # Ruta de la carpeta de entrada
    input_folder = args.path if args.path else '.'

    # Crear una lista para almacenar las rutas de las imágenes
    image_paths = []

    # Recorrer todas las imágenes en la carpeta de entrada y sus subcarpetas
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):  # Asegúrate de que es una imagen
                image_paths.append(os.path.join(root, file))

    # Crear un ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        # Convertir las imágenes en paralelo
        executor.map(convert_to_png, image_paths)

if __name__ == '__main__':
    main()
