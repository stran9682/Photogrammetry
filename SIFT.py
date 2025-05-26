import cv2
import numpy as np

# Generates an octave of images with progressively more Gaussian blur applied
# Gaussians progressively increase by a factor of k = 2^(1/s)
# Where 's' is the number of intervals such that the Guassian Blur effect doubles
# s+3 = 5, with an inital sigma of 1.6 was chosen to reflect OpenCV's implementation
def generateOctave (baseImage) :

    prevSigma = 1.6
    intervals = 2
    k = (2**(1/intervals))

    base = cv2.GaussianBlur(baseImage, (0,0), prevSigma)

    images = list() 
    images.append(base)

    for i in range(intervals+2):
        
        sigmaTotal = prevSigma * k

        appliedBlur = np.sqrt(sigmaTotal**2 - prevSigma**2)

        image = cv2.GaussianBlur(images[i], (0,0), appliedBlur)

        images.append(image)

        prevSigma = sigmaTotal

    return images

def generatePyramid (baseImage) :

    numOctaves = 4

    images = list()

    for i in range(numOctaves):
        images.append(generateOctave(baseImage))


image = cv2.imread("kitty.png")
generateOctave(image)




    


    

