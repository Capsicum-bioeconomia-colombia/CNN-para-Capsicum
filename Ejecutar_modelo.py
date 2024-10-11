#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:48:53 2024

@author: javiermoreno
"""
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Poner la ruta donde se guarda la carpeta
workdir = "/Users/javiermoreno/Documents/Agrosavia_Capsicum_2023/Entregables/Modelos/Red_neuronal_convolucional/"

# Carga el modelo
model = load_model(workdir + 'FlorFruto_modelo_v5_FF_final.h5')

# Path de la carpeta de imágenes
image_folder = workdir + "imagenes/"

labels = {0: "Annuum", 1: "Baccatum", 2: "Chinnense", 3: "Frutescens", 4: "Pubescens"}


# Define la función que procesará la imagen y generará la clasificación
def process_image(image):
    # Redimensiona la imagen a las dimensiones adecuadas para el modelo
    image = image.resize((224, 224))
    # Convierte la imagen a un arreglo de numpy
    image_array = np.array(image)
    # Normaliza los valores de los pixeles
    image_array = image_array / 255.0
    # Agrega una dimensión adicional para que el modelo pueda procesar la imagen
    image_array = np.expand_dims(image_array, axis=0)
    # Utiliza el modelo para generar la clasificación
    predictions = model.predict(image_array)
    # Obtiene la clasificación con mayor probabilidad
    classification = np.argmax(predictions[0])
    return int(classification)


# Iterar sobre todos los archivos en la carpeta de imágenes
mensajes = []
for filename in os.listdir(image_folder):
    # Verificar si el archivo es una imagen
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.PNG', '.JPEG')):
        cont += 1
        # Path de la imagen
        image_path = os.path.join(image_folder, filename)

        # Leer la imagen utilizando PIL
        img = Image.open(image_path)
        label = labels[process_image(img)]
        mensaje = f"La imagen {filename} es {label}\n"
        print(mensaje)
        mensajes.append(mensaje)

print(correct/cont)

with open(workdir + 'imagenes_procesadas.txt', 'w') as f:
    f.writelines(mensajes)
