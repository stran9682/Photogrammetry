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

    return DoGs

def getLocalExtrema (DoG_Octave):

    octave_extrema = list()

    for i in range (1, len(DoG_Octave) -1):

        dog_prev = DoG_Octave[i - 1]
        dog_curr = DoG_Octave[i]
        dog_next = DoG_Octave[i + 1]

        octave_extrema.append(findExtrema (dog_prev, dog_curr, dog_next))

    return octave_extrema

def findExtrema (DoG_prev, DoG_curr, DoG_next):

    extremas = list()

    for row in range(1, DoG_curr.shape[0]-1):

        for col in range (1, DoG_curr.shape[1]-1):

            location = (row, col)

            possibleExtrema = checkNeighbors (DoG_prev, DoG_curr, DoG_next, location)
            
            if possibleExtrema:
                extremas.append(possibleExtrema)

    print("Done image, found " + str(len(extremas)) + " extrema")

    return extremas

def checkNeighbors (DoG_prev, DoG_curr, DoG_next, location) : 

    x = location[1]
    y = location[0]

    patch_prev = DoG_prev[y-1:y+2, x-1:x+2]
    patch_curr = DoG_curr[y-1:y+2, x-1:x+2]
    patch_next = DoG_next[y-1:y+2, x-1:x+2]

    neighbors = np.concatenate([
        patch_prev.flatten(),
        patch_curr.flatten(),
        patch_next.flatten()
    ])

    center = neighbors[13]

    neighbors = np.delete(neighbors, 13)

    if center < np.min(neighbors) or center > np.max(neighbors):
        return center
    
    return None

image = cv2.imread("kitty.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
octaves = generateOctaves(image)

for octave in octaves:
    DoGS = createDifferenceOfGaussians(octave)

    print("new level!")

    extrema = getLocalExtrema (DoGS)