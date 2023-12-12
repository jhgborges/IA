import cv2
import numpy as np

def calculate_histogram(image):
    # Convertendo a imagem para tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculando o histograma da imagem
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalizando o histograma
    hist = hist / hist.sum()

    return hist

def calculate_image_similarity(image1, image2):
    # Calculando histogramas para ambas as imagens
    hist1 = calculate_histogram(image1)
    hist2 = calculate_histogram(image2)

    # Calculando a similaridade usando a correlação de histogramas
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Convertendo a similaridade para porcentagem
    similarity_percentage = (similarity + 1) * 50

    return similarity_percentage

# Carregando as imagens
image_path1 = "img1.jpeg"
image_path2 = "img2.jpeg"

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# Calculando a similaridade entre as imagens
similarity = calculate_image_similarity(image1, image2)

# Exibindo a similaridade em porcentagem
print(f"A similaridade entre as imagens é: {similarity:.2f}%")

