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
sobelX = np.matrix('-1 0 1; -2 0 2; -1 0 1')
sobelY = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')
promedioPixel = 0
promedioSobelX = 0
promedioSobelY = 0


print ("{}".format(kernelPromedio))
print ("{}".format(sobelX))
print ("{}".format(sobelY))
cv2.imshow("Ponderada", imagePonderada)

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


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imagenBorrosa = cv2.GaussianBlur (gray, (3,3), 0)
#cv2.imshow("Blurred", imagenBorrosa)



imagenSobelX = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 1, 0)
imagenSobelY = cv2.Sobel(imagenBorrosa, cv2.CV_64F, 0, 1)

sobelCombined = cv2.bitwise_or(imagenSobelX, imagenSobelY)

plt.subplot(2,2,1),plt.imshow(imagenSobelX, cmap='gray')
plt.subplot(2,2,2),plt.imshow(imagenSobelY, cmap='gray')
plt.subplot(2,2,3),plt.imshow( sobelCombined, cmap='gray')

plt.show()


cv2.waitKey(0)