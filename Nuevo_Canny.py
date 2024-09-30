import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
original = cv2.imread('C:/Users/manue/Desktop/imgs/6.jpg')
original=cv2.resize(original,(300,300))
image=original.copy()
image=cv2.resize(image,(300,300))

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.title('Distribución de Píxeles en la Imagen en Escala de Grises')
plt.xlabel('Valor de Píxel')
plt.ylabel('Frecuencia')
plt.show()


#Umbralización, blanco y negro, imagen binaria
umbral = 86
_, mosquitos_negros = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

# Invertir los colores para tener los mosquitos en negro
mosquitos_blancos = cv2.bitwise_not(mosquitos_negros)

# Aplicar dilatación para ajustar la forma y el tamaño de los mosquitos
kernel_dilate = np.ones((3, 3), np.uint8)
mosquitos_dilate = cv2.dilate(mosquitos_blancos, kernel_dilate, iterations=2)

# Aplicar el algoritmo de detección de bordes Canny
edges = cv2.Canny(mosquitos_dilate, umbral, 255)

# Encontrar contornos en la imagen de bordes
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos encontrados en la imagen original
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

# Mostrar la imagen original con los contornos dibujados
cv2.imshow('Contours', image)
cv2.imshow('Original', original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contar y mostrar el número de objetos encontrados
print(f'Se encontraron {len(contours)} objetos en la imagen.')
