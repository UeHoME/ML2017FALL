import numpy as np
from skimage import io
from os import listdir
import sys
from skimage import transform

def loadImages(path):
    loadedImages = []
    for i in range(415):
        imgpath = path + str(i) + '.jpg'
        img = io.imread(imgpath)
        # img = transform.resize(img, (600, 600, 3))
        loadedImages.append(img.flatten())
    return loadedImages

def normalize(matrix):
    matrix = matrix.astype(np.float32)
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    matrix = np.array((matrix * 255).astype(np.uint8))
    return matrix

def get_eigenface(e_vector, eigenvalue, number = 1):
    print("Show image")
    for nb in range(number):
        eigenface = e_vector[:,nb]
        img = normalize(eigenface)
        s = 'eigenface_' + str(nb) + '.jpg'
        print(s)
        io.imsave(s,img.reshape(600,600,3))

def reconstruction(eigenface, target, mean_face):
    target = target - mean_face
    w = np.dot(eigenface.T, target)
    reconstruct = (eigenface.dot(w)).reshape(600, 600, 3) + mean_face.reshape(600, 600, 3)
    reconstruct = normalize(reconstruct)
    # reconstruct = transform.resize(reconstruct, (600, 600, 3))
    io.imsave('reconstruction.jpg',reconstruct)

def main():
    path = sys.argv[1]
    imgs = loadImages(path)
    image = np.asarray(imgs).T

    mean_face = np.mean(image,axis = 1)

    e_vector,e_value,V = np.linalg.svd(image, full_matrices = False) # eigenvector 270000 x 415
    e_sum = np.sum(e_value)
    # for nb in range(4):
    #     s = "w" + str(nb + 1) + '='
    #     print(s, e_value[nb] / e_sum)
    print("Done SVM")
    io.imsave('eigenface_10.jpg', normalize(e_vector[:,9]).reshape(600, 600, 3))
    print(e_vector.shape) 
    get_eigenface(e_vector, e_value, number = 4)
    eigenface = e_vector[:, 0:4]
    img = image[:, int(sys.argv[2].split('.')[0])]
    reconstruction(eigenface, img.reshape(600 * 600 * 3,), mean_face)
if __name__ == "__main__":
    main()