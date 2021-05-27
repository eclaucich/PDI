import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def about():
    print("Funciones relaciones con el procesamiento de histogramas")

#HISTOGRAMAS
def hist_normal(img):
    """
    Calcula el histograma de la imagen
    """
    img = img.astype('uint8')
    return cv.calcHist([img], [0], None, [256], [0,256])

def eq_global(img):
    """
    Calcula el histograma ecualizado globalmente de la imagen
    """
    img = img.astype('uint8')
    return cv.equalizeHist(img)

def eq_local(img):
    """
    Calcula el histograma ecualizado localmente (CLAHE) de la imagen
    """
    img = img.astype('uint8')
    clahe = cv.createCLAHE()
    cl = clahe.apply(img)
    return cl


def plot_histo(imgs, figsize=(20,5), patron=(1,2), gray=True, eq='normal', titulos=[], vmin=0, vmax=255):
    """
    Dibuja una imagen y su histograma

    @param eq:  especifica la ecualización del histograma:\n \'normal\' -> sin ecualización \n\'global\' -> ecualización global \n \'local\' -> ecualización local'
    """
    assert patron[0]*patron[1] >= len(imgs)*2, 'ERROR en el patrón del subplot.\n Verifica que el parámetro \'patron\' contemple la cantidad suficiente de imágenes'

    assert len(titulos)==0 or len(imgs)*2==len(titulos), 'ERROR la cantidad de imágenes debe ser igual a la cantidad de títulos'

    assert eq == 'normal' or eq == 'global' or eq == 'local', 'ERROR en el tipo de ecualización.'

    plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        plt.subplot(patron[0], patron[1], (2*i)+1)
        if(len(titulos)>0): 
            plt.title(titulos[(2*i)])

        if eq=='global':
            img = eq_global(img)
        elif eq=='local':
            img = eq_local(img)

        if(gray):
            plt.imshow(img.astype('uint8'), cmap='gray', vmin=vmin, vmax=vmax)
        else:
            plt.imshow(img.astype('uint8'), vmin=vmin, vmax=vmax)

        plt.subplot(patron[0], patron[1], (2*i)+2)
        if(len(titulos)>0): 
            plt.title(titulos[(2*i)+1])
        plt.hist((img.astype('uint8')).ravel(), 256, [0,256])

    
#ESTADSITICAS HISTOGRAMAS
def probOcurrencia(img):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    accum = np.sum(hist[:,0])
    return hist/accum

def media(img):
    return np.dot(np.arange(256),probOcurrencia(img))

def varianza(img):
    return np.dot(((np.arange(256)-media(img))**2), probOcurrencia(img))

def energia(img):
    return np.sum(probOcurrencia(img)**2)

def asimetria(img):
    return np.dot(((np.arange(256)-media(img))**3), probOcurrencia(img))

def entropia(img):
    img_log = np.log2(probOcurrencia(img).clip(min=0.00001))
    return -np.sum(probOcurrencia(img)*img_log)