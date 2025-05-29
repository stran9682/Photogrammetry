import cv2
import numpy as np

# Generates an octave of s+3 images with progressively more Gaussian blur applied
# Gaussians progressively increase by a factor of k = 2^(1/s)
# Where 's' is the number of intervals such that the Guassian Blur effect doubles
# +3 is to make sure that there enough DoGs produced to cover the entire octave
# Here s+3 = 5, with an inital sigma of 1.6, chosen to reflect OpenCV's implementation
# However, David Lowe's paper suggests using s+3 = 6
def generateOctave (base_Blurred_Image, prev_Sigma, intervals) :

    k = (2**(1/intervals))

    images = list() 
    images.append(base_Blurred_Image)

    for i in range(intervals+2):

        sigmaTotal = prev_Sigma * k

        appliedBlur = np.sqrt(sigmaTotal**2 - prev_Sigma**2)

        image = cv2.GaussianBlur(images[i], (0,0), appliedBlur)

        images.append(image)

        prev_Sigma = sigmaTotal

    return images

# Generating multiple octaves of blurred images
# the next octave starts at image with 2 sigma from the base, then downsampled by half
# ensuring scale space continuity
def generateOctaves (image, iniital_sigma = 1.6, intervals = 2) :

    octaves = list()

    baseImage = cv2.GaussianBlur(image, (0,0), 1.6)

    octaves.append(generateOctave(baseImage, iniital_sigma, intervals))

    for i in range(1,4):

        doubleSigmaImage = octaves[-1][-3]

        baseImage = cv2.resize(
            doubleSigmaImage, 
            (int(doubleSigmaImage.shape[1] / 2), int(doubleSigmaImage.shape[0] / 2)), 
            interpolation=cv2.INTER_LINEAR)

        octave = generateOctave(baseImage, iniital_sigma, intervals)

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

        cv2.imshow("image", DoG)
        cv2.waitKey(0)
        cv2.destroyAllWindow

def detectLocalExtrema (DoG_Images):

    for i in range (1, len(DoG_Images) -1):
        dog_prev = DoG_Images[i - 1]
        dog_curr = DoG_Images[i]
        dog_next = DoG_Images[i + 1]

        # TODO loop through pixels of current and provide location to check neighbros

        checkNeighbors(dog_prev, dog_curr, dog_next)

def checkNeighbors (DoG_prev, DoG_curr, DoG_next, location) : 

    pixels = list()

    for row in range (-1, 2):
        for col in range (-1, 2):
            pixels.append(DoG_prev[location[0] + row][location[1] + col])
            pixels.append(DoG_curr[location[0] + row][location[1] + col])
            pixels.append(DoG_next[location[0] + row][location[1] + col])

    center = pixels.pop(13)

    if center == min(pixels) or center == max(pixels):
        return True
    
    return False

image = cv2.imread("kitty.png")
octaves = generateOctaves(image)

for octave in octaves:
    createDifferenceOfGaussians(octave)





    


    

