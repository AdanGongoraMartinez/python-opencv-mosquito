import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('C:/Users/manue/Desktop/imgs/6.jpg')
image=cv2.resize(image,(300,300))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Umbralización, blanco y negro, imagen binaria
umbral = 86
_, mosquitos_negros = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

# Invertir los colores para tener los mosquitos en negro
mosquitos_blancos = cv2.bitwise_not(mosquitos_negros)

# Aplicar dilatación para ajustar la forma y el tamaño de los mosquitos
kernel_dilate = np.ones((3, 3), np.uint8)
mosquitos_dilate = cv2.dilate(mosquitos_blancos, kernel_dilate, iterations=2)



# Aplicar el filtro Sobel para detectar bordes
sobelx = cv2.Sobel(mosquitos_dilate, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(mosquitos_dilate, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(sobel_combined)

# Umbralización para convertir la imagen a binaria
_, binary = cv2.threshold(sobel_combined, 200, 255, cv2.THRESH_BINARY)

# Encontrar contornos
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen original
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

# Mostrar el número de objetos detectados
print(f'Número de objetos detectados: {len(contours)}')

# Mostrar las imágenes
cv2.imshow('Imagen original', image)
#cv2.imshow('Sobel Combined', sobel_combined)
#cv2.imshow('Binaria', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
