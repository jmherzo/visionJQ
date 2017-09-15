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
tamanoKernel = 5
factorSaturacion = 273

#Imprimimos el valor de ancho y alto de la imagen
print ("width:{}pixels".format(image.shape[1]))
imageWidth = image.shape[1]
print ("height:{}pixels".format(image.shape[0]))
imageHeight = image.shape[0]

#cv2.imshow("Original", image)

kernelPromedio = np.matrix('1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7; 4 16 26 16 4; 1 4 7 4 1')
promedioPixel = 0

#Convertimos imagen original en blanco y neggro
for y in range(0, imageWidth):
	for x in range(0, imageHeight):
		(b, g, r) = image[x,y]
		ponderado = 0.21*r + 0.72*g + 0.07*b
		intPonderado = int (ponderado)
		imagePonderada[x,y]=(ponderado, ponderado, ponderado)


print ("{}".format(kernelPromedio))
cv2.imshow("Ponderada", imagePonderada)
cv2.imshow("Libreria", cv2.GaussianBlur(image,(tamanoKernel,tamanoKernel), 0))

imagenBorrosa = image
margin = int(tamanoKernel/2)
#print("Margin: {}".format(margin))
ancho = imageWidth-tamanoKernel
alto = imageHeight- tamanoKernel


#Empezamos a recorrer los pixeles de cada imagen para el proceso de convolucion
for y in range(margin, ancho):
	for x in range(margin, alto):
		for z in range (0, tamanoKernel-1):
			for w in range(0, tamanoKernel-1):
				(b,g,r) = imagePonderada[x+w,y+z]
				promedioPixel  = promedioPixel + (b * kernelPromedio[w,z])
		promedioPixel = promedioPixel / factorSaturacion 
		imagenBorrosa[x,y] = promedioPixel
		promedioPixel = 0

cv2.imshow("Blurred", imagenBorrosa)


cv2.waitKey(0)