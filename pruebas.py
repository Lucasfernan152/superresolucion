import cv2
from matplotlib import pyplot as plt
from procesar_imagenes import *

imagen = cv2.imread('Recortes\imagen0.jpg')
subimagen = imagen[0:3,0:3,0]
print(subimagen)
print(mediaPonderada(subimagen, 3))
