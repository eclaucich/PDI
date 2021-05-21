import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def about():
    print('Funciones refereidas al procesamiento de color de imágenes')


def get_intensidad(img):
    """
    Calcula el canal de intensidad de una imagen.\n
    Se supone una imagen en RGB
    """
    [r,g,b] = cv.split(img)
    return ((r+g+b)/3).astype('uint8')


def plot_color_channels(img, figsize=(20,15), patron=(1,4), space='rgb'):
    """
    Dibuja la imagen original y los distintos canales de la misma.\n
    La imagen debe estar en RGB.
    
    @param space: espacio de los colores de la imagen.\n    rgb/hsv/hsi
    """

    assert space=='rgb' or space=='hsv' or space=='hsi', 'ERROR, atributo \'space\' incorrecto'

    plt.figure(figsize=figsize)
    plt.subplot(patron[0],patron[1],1)
    plt.title('ORIGINAL')
    plt.imshow(img)
    
    channel_names = ['R', 'G', 'B']
    channels = cv.split(img)
    
    if(space=='hsv'):
        channel_names = ['H', 'S', 'V']
        channels = cv.split(RGB2HSV(img))

    elif(space=='hsi'):
        channel_names = ['H', 'S', 'I']
        channels = cv.split(RGB2HSI(img))

    plt.subplot(patron[0],patron[1],2)
    plt.title(channel_names[0])
    plt.imshow(channels[0], cmap='gray')
    
    plt.subplot(patron[0],patron[1],3)
    plt.title(channel_names[1])
    plt.imshow(channels[1], cmap='gray')
    
    plt.subplot(patron[0],patron[1],4)
    plt.title(channel_names[2])
    plt.imshow(channels[2], cmap='gray')


def RGB2HSV(img):
    return cv.cvtColor(img, cv.COLOR_RGB2HSV)

def RGB2HSI(img):
    h,s,_ = cv.split(RGB2HSV(img))
    i = get_intensidad(img)
    
    return cv.merge([h,s,i])

def HSV2RGB(img):
    return cv.cvtColor(img, cv.COLOR_HSV2RGB)

def HSV2HSI(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    r,g,b = cv.split(img_rgb)
    i = get_intensidad(img_rgb)
    h,s,_ = cv.split(img)

    return cv.merge([h,s,i])

def HSI2RGB(img):
    pass


def HSI2HSV(img):
    pass


def color_slicing_sphere_RGB(img, centre, r):
    """
    Realiza la segmentación basada en color, mediante una esfera centrada en el color \'centre\' de radio \'r\'

    @param centre: centro de la esfera de la forma (R, G, B)
    @param r: radio de la esfera
    @return numpy array con los índices que caen por fuera de la esfera
    """

    assert r>=0, 'ERROR, el radio debe ser positivo'
    
    #normalizo los valores de la imagen y me fijo en qué índices los valores estan por fuera de la esfera
    idx_fuera = np.where(np.sum(((img/255) - centre)**2, axis=2) > r**2)
    return idx_fuera