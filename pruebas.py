import cv2
from matplotlib import pyplot as plt
from procesar_imagenes import *

image = cv2.imread("PDI\Recortes\imagen0.jpg")
print(image.shape)
nueva = ecualizarImagenes_Local(image)

fig = plt.figure()
fig.suptitle("Imagenes")
h1 = plt.subplot(2,3,2), plt.imshow(image), plt.title('Imagen Original')
h2 = plt.subplot(2,3,4), plt.imshow(nueva), plt.title('Ecualizada Localmente')
h2 = plt.subplot(2,3,5), plt.imshow(image-nueva), plt.title('Ecualizada Localmente')
plt.show()