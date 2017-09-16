from __future__ import print_function
from __future__ import division
import cv2
import argparse
from matplotlib import pyplot as plt
import numpy as np
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image")
args = vars(ap.parse_args())
imageName = args["image"]
print ("name:{}".format(imageName))
rowsPlot = 2
colsPlot = 4


image = cv2.imread(args["image"])


tamanoKernel = 5
factorSaturacion = 273
#Imprimimos el valor de ancho y alto de la imagen
print ("width:{}pixels".format(image.shape[1]))
imageWidth = image.shape[1]
print ("height:{}pixels".format(image.shape[0]))
imageHeight = image.shape[0]

cv2.imshow("Original", image)
plt.subplot(rowsPlot,colsPlot,2),plt.imshow(image)
plt.title('Original'), plt.xticks([]), plt.yticks([])

kernelPromedio = np.matrix('1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7; 4 16 26 16 4; 1 4 7 4 1')
#sobelX = np.matrix('-2 0 2; -6 0 6; -2 0 2')
#sobelY = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')
sobelX = np.matrix('-3 0 3; -10 0 10; -3 0 3')
sobelY = np.matrix('-3 -10 -3; 0 0 0; 3 10 3')
#sobelY = np.matrix('0 0 0; 0 2 0; 0 0 0')
promedioPixel = 0
promedioSobelX = 0
promedioSobelY = 0
margin = int(tamanoKernel/2)
ancho = imageWidth-tamanoKernel
alto = imageHeight- tamanoKernel
print ("{}".format(kernelPromedio))
plt.subplot(rowsPlot,colsPlot,1),plt.imshow(kernelPromedio, cmap = 'gray')
plt.title('Kernel'), plt.xticks([]), plt.yticks([])

imagePonderada = np.zeros(shape=(imageHeight,imageWidth), dtype=np.float)
for y in range(0, imageHeight-1):
	for x in range(0, imageWidth-1):
		(bl,gr,re) = image[y,x]
		imagePonderada[y,x] = float(re) / 255
plt.subplot(rowsPlot,colsPlot,3),plt.imshow(imagePonderada, cmap = 'gray')
plt.title('Ponderada'), plt.xticks([]), plt.yticks([])
print ("{}".format(imagePonderada))

imagenBorrosa = np.zeros(shape=(imageHeight,imageWidth), dtype=np.float)
#Empezamos a recorrer los pixeles de cada imagen para el proceso de convolucion
for y in range(margin, ancho):
	for x in range(margin, alto):
		for z in range (0, tamanoKernel-1):
			for w in range(0, tamanoKernel-1):
				temp = imagePonderada[x+w,y+z]
				promedioPixel  = promedioPixel + (temp * kernelPromedio[w,z])
		promedioPixel = promedioPixel / factorSaturacion 
		imagenBorrosa[x,y] = promedioPixel
		promedioPixel = 0
plt.subplot(rowsPlot,colsPlot,4),plt.imshow(imagenBorrosa, cmap = 'gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])
imagenTemp = imagenBorrosa.copy()

for y in range(0, imageHeight-1):
	for x in range(0, imageWidth-1):
		imagenTemp[y,x] = imagePonderada[y,x] - imagenBorrosa [y,x]
		
plt.subplot(rowsPlot,colsPlot,5),plt.imshow(imagenTemp, cmap = 'gray')
plt.title('Ponderada - Blurred'), plt.xticks([]), plt.yticks([])

imagenSobelX = imagePonderada.copy()
imagenSobelY = imagePonderada.copy()

#Imagen a la que le aplicamos las mascaras de Sobel
imagenTemp2 = imagenTemp.copy()


#Aplicamos mascaras de Sobel
for c in range(1, imageHeight-1):
	for n in range(1, imageWidth-1):
		for m in range (0, 2):
			for v in range(0, 2):
				tempo = imagenTemp2[c+v,n+m]
				promedioSobelX  = promedioSobelX + (tempo * sobelX[v,m])
				promedioSobelY  = promedioSobelY + (tempo * sobelY[v,m])
		imagenSobelX[c,n] = promedioSobelX
		imagenSobelY[c,n] = promedioSobelY
		promedioSobelX = 0
		promedioSobelY = 0
plt.subplot(rowsPlot,colsPlot,6),plt.imshow(imagenSobelX, cmap = 'gray')
plt.title('SobelX'), plt.xticks([]), plt.yticks([])
plt.subplot(rowsPlot,colsPlot,7),plt.imshow(imagenSobelY, cmap = 'gray')
plt.title('SobelY'), plt.xticks([]), plt.yticks([])

sobelCombined = imagePonderada.copy()
#Sacamos la magnitud del gradiente con la raiz de la suma de los cuadrados
for y in range(0, imageHeight-1):
	for x in range(0, imageWidth-1):
		sobelCombined[y,x] = math.sqrt((imagenSobelY[y,x] ** 2 )+(imagenSobelX[y,x]**2)) 
#sobelCombined = cv2.bitwise_or(imagenSobelX, imagenSobelY)
plt.subplot(rowsPlot,colsPlot,8),plt.imshow(sobelCombined, cmap = 'gray')
plt.title('SobelCombined'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)