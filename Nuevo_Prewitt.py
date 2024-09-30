import cv2
import numpy as np

# Función para aplicar el operador Prewitt a una imagen en escala de grises
def prewitt_edge_detection(image):
    # Convertir la imagen a escala de grises si no lo está
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Aplicar el operador Prewitt
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])

    prewitt_x = cv2.filter2D(gray_image, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray_image, -1, kernel_y)

    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    return magnitude

# Función para contar el número de objetos en una imagen binaria
def count_objects(binary_image):
    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# Cargar la imagen
image_path = 'C:/Users/manue/Desktop/imgs/6.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image,(300,300))

umbral = 86
_, mosquitos_negros = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)

# Invertir los colores para tener los mosquitos en negro
mosquitos_blancos = cv2.bitwise_not(mosquitos_negros)

# Aplicar dilatación para ajustar la forma y el tamaño de los mosquitos
kernel_dilate = np.ones((3, 3), np.uint8)
mosquitos_dilate = cv2.dilate(mosquitos_blancos, kernel_dilate, iterations=2)

# Aplicar detección de bordes con el operador Prewitt
edges = prewitt_edge_detection(mosquitos_dilate)

# Convertir la imagen de bordes a un tipo de datos compatible
edges_uint8 = np.uint8(edges)

# Binarizar los bordes con un umbral más bajo
blur = cv2.GaussianBlur(edges_uint8,(3,3),0)
_, binary_edges = cv2.threshold(edges_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplicar una operación de dilatación para conectar los bordes
kernel = np.ones((3,3), np.uint8)
binary_edges_dilated = cv2.dilate(binary_edges, kernel, iterations=1)

# Contar el número de objetos
num_objects = count_objects(binary_edges_dilated)

# Mostrar la imagen original y los bordes binarizados
cv2.imshow('Original Image', image)
cv2.imshow('Binarized Edges', binary_edges_dilated)
print("Número de objetos detectados:", num_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()