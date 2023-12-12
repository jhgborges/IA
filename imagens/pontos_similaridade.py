import cv2
import numpy as np

def find_and_draw_matches(img1, img2):
    # Inicializa o detector ORB
    orb = cv2.ORB_create()

    # Encontra os pontos-chave e descritores com o ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Cria o objeto correspondente de força-bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Faz a correspondência dos descritores
    matches = bf.match(des1, des2)

    # Ordena os matches com base nas distâncias
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenha os matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Exibe a imagem com os matches
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Carrega as imagens
#image_path1 = "img1.jpeg"
#image_path2 = "img2.jpeg"
image_path1 = "img3.png"
image_path2 = "img4.png"

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# Converte as imagens para tons de cinza
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Encontra e desenha os matches
find_and_draw_matches(gray_image1, gray_image2)
