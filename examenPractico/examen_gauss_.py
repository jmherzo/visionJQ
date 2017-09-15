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
imagePonderada = image.copy()

tamanoKernel = 5
factorSaturacion = 273

#Imprimimos el valor de ancho y alto de la imagen
print ("width:{}pixels".format(image.shape[1]))
imageWidth = image.shape[1]
print ("height:{}pixels".format(image.shape[0]))
imageHeight = image.shape[0]

#cv2.imshow("Original", image)

kernelPromedio = np.matrix('1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7; 4 16 26 16 4; 1 4 7 4 1')
sobelX = np.matrix('-1 0 1; -2 0 2; -1 0 1')
sobelY = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')
promedioPixel = 0
promedioSobelX = 0
promedioSobelY = 0


print ("{}".format(kernelPromedio))
print ("{}".format(sobelX))
print ("{}".format(sobelY))
cv2.imshow("Ponderada", imagePonderada)

imagenBorrosa = image.copy()
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


plt.subplot(2,2,1),plt.imshow(imagenBorrosa)
plt.title('Blurred'), plt.xticks([]), plt.yticks([])

imagenSobelX = imagenBorrosa.copy()
#Aplicamos mascaras de Sobel
for c in range(1, imageHeight-3):
	for n in range(1, imageWidth-3):
		for m in range (0, 2):
			for v in range(0, 2):
				(b,g,r) = imagenBorrosa[c+v,n+m]
				promedioSobelX  = promedioSobelX + (b * sobelX[v,m])
		imagenSobelX[c,n] = promedioSobelX
		promedioSobelX = 0
plt.subplot(2,2,3),plt.imshow(imagenSobelX)
plt.title('SobelX'), plt.xticks([]), plt.yticks([])
#cv2.imshow("SobelX", imagenSobelX)


imagenSobelY = imagenBorrosa.copy()
#Aplicamos mascaras de Sobel
for t in range(1, imageHeight-3):
	for f in range(1, imageWidth-3):
		for p in range (0, 2):
			for o in range(0, 2):
				(b,g,r) = imagenBorrosa[t+o,f+p]
				promedioSobelY  = promedioSobelY + (g * sobelY[o,p])
		imagenSobelY[t,f] = promedioSobelY
		promedioSobelY = 0
plt.subplot(2,2,2),plt.imshow(imagenSobelY)
plt.title('SobelY'), plt.xticks([]), plt.yticks([])
#cv2.imshow("SobelY", imagenSobelY)


sobelCombined = cv2.bitwise_or(imagenSobelX, imagenSobelY)
plt.subplot(2,2,4),plt.imshow(sobelCombined)
plt.title('SobelCombined'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)