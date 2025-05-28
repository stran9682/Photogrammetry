import cv2
import numpy as np

# Generates an octave of s+3 images with progressively more Gaussian blur applied
# Gaussians progressively increase by a factor of k = 2^(1/s)
# Where 's' is the number of intervals such that the Guassian Blur effect doubles
# +3 is to make sure that there enough DoGs produced to cover the entire octave
# Here s+3 = 5, with an inital sigma of 1.6, chosen to reflect OpenCV's implementation
# However, David Lowe's paper suggests using s+3 = 6
def generateOctave (baseImage, prevSigma = 1.6, intervals = 2) :

    k = (2**(1/intervals))

    images = list() 
    images.append(baseImage)

    for i in range(intervals+2):

        sigmaTotal = prevSigma * k

        appliedBlur = np.sqrt(sigmaTotal**2 - prevSigma**2)

        image = cv2.GaussianBlur(images[i], (0,0), appliedBlur)

        images.append(image)

        prevSigma = sigmaTotal

    return images

def generateOctaves (image) :

    octaves = list()

    baseImage = cv2.GaussianBlur(image, (0,0), 1.6)

    octaves.append(generateOctave(baseImage))

    for i in range(1,4):

        doubleSigmaImage = octaves[-1][-3]

        baseImage = cv2.resize(
            doubleSigmaImage, 
            (int(doubleSigmaImage.shape[1] / 2), int(doubleSigmaImage.shape[0] / 2)), 
            interpolation=cv2.INTER_LINEAR)

        octave = generateOctave(baseImage)

        octaves.append(octave)

    return octaves
    
# Subtracts a less blurred image from another more blurred image
# Highlights the edges by highlighting the changes that occur between blur levels
# A lower sigma means more detail is saved, while a higher sigma will discard more
# A high difference will indicate an edge
def createDifferenceOfGaussians (octave):

    DoGs = list()

    for i in range(1, len(octave)):

        DoG = cv2.subtract(octave[i], octave[i-1])
        DoGs.append(DoG)

def detectLocalExtrema (DoG_Images):

    for i in range (1, len(DoG_Images) -1):
        dog_prev = DoG_Images[i - 1]
        dog_curr = DoG_Images[i]
        dog_next = DoG_Images[i + 1]

image = cv2.imread("kitty.png")
octave = generateOctaves(image)





    


    

