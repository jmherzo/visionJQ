from __future__ import print_function
from __future__ import division

import cv2
import argparse
from matplotlib import pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image")
args = vars(ap.parse_args())
imageName = args["image"]
print ("name:{}".format(imageName))



image = cv2.imread(args["image"])
imagePonderada = image

#Imprimimos el valor de ancho y alto de la imagen
print ("width:{}pixels".format(image.shape[1]))
imageWidth = image.shape[1]
print ("height:{}pixels".format(image.shape[0]))
imageHeight = image.shape[0]

cv2.imshow("Original", image)
hist = np.zeros((256,), dtype=np.int)
histNorm = np.zeros((256,), dtype=np.int)
maximo = 0
minimo = 256
maaximoNuevo = 0
minimoNuevo = 0

#Empezamos a recorrer los pixeles de cada imagen
for y in range(0, imageWidth):
	for x in range(0, imageHeight):
		#Obtenemos los valores RGB del pixel 
		(b, g, r) = image[x,y]
		#Realizamos los calculos necesarios para cada tipo de conversion
		ponderado = 0.21*r + 0.72*g + 0.07*b
		intPonderado = int (ponderado)
		hist [intPonderado] = hist[intPonderado] + 1
		if (intPonderado > maximo) and (intPonderado < 256):
			maximo = intPonderado
		if (intPonderado < minimo) and (intPonderado > 0):
			minimo = intPonderado
		#Asignamos los valores calculados al pixel en la imagen final respectiva
		imagePonderada[x,y]=(ponderado, ponderado, ponderado)
		#print ("{}".format(intPonderado), end=" ")
	#print ("")
#imagenNormalizada = imagePonderada
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("Density of pixels")
plt.plot(hist)
plt.xlim ([0,256])
plt.savefig("Histograma:{}".format(imageName))
cv2.imshow("Ponderada", imagePonderada)

print ("Maximo:{}".format(maximo))
print ("Minimo:{}".format(minimo))
maaximoNuevo = 255
minimoNuevo = 0
print ("Maximo nuevo:{}".format(maaximoNuevo))
print ("Minimo nuevo:{}".format(minimoNuevo))

imagenNormalizada = imagePonderada

factor = (maaximoNuevo-minimoNuevo)/(maximo-minimo)
print ("Factor:{}".format(factor))
for k in range (0, imageWidth):
	for l in range (0, imageHeight):
		imagenNormalizada [l,k] = (factor) * (imagenNormalizada[l,k] - minimo) - minimoNuevo

#Mostramos cada imagen en una ventana diferente

for y in range(0, imageWidth):
	for x in range(0, imageHeight):
		valor = imagenNormalizada[x,y]
		histNorm [valor] = histNorm[valor] + 1
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("Density of pixels")
plt.plot(histNorm)
plt.xlim ([0,256])
plt.savefig("HistNormalizado:{}".format(imageName))
cv2.imshow("Normalizada", imagenNormalizada)
#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow("B&W", image)

#hist = cv2.calcHist([image], [0], None, [256], [0, 256])

cv2.waitKey(0)