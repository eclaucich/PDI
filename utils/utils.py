import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def about():
    print('Funciones varias para PDI')


##TRANSFORMACIONES
### La idea es que si tenemos una imagen 'img' a la que queramos aplicar una transformación 'T' hacemos:
#### t_T[img] para aplicar la trasnformación a la imagen

def t_lut(a, c):
    """
    Devuelve una transformación LUT con la forma:
    a * x + c
    """
    
    dom = np.array(np.arange(256))
    lut = a*dom + c
    return lut

def t_lut_tramos(params):
    """
    Aplica una transformación LUT definida por tramos.

    Recibe una lista con N tuplas de la siguiente forma:

    [(l0, a0, c0), (l1, a1, c1), ... (a_n-1, c_n-1)]

    El primer tramo va de 0 a l0 y el último va de l(n-2) al máximo
    de intensidad del canal.

    l_i: Limite superior del tramo
    a_i: Ganancia en el tramo (pendiente)
    c_i: Offset en el tramo (ordenada al origen)

    Nota: La última tupla de la lista tiene solo dos elementos, ya que el
    limite superior es siempre el valor maximo de intensidad del canal (255)

    Funciona solo si `params` tiene al menos dos elementos!

    Ejemplo:
    
    Tramo de 0 a 100 no hace nada (funcion identidad) y de ahí en adelante
    es la funcion constante s = 255.

    t_lut_tramos([100, 1, 0], [0, 255])
    """
    dom = np.array(np.arange(256))

    def lut(a, c):
        return lambda x: a * x + c

    condlist = []
    funclist = []

    # Tupla por tupla definimos una mascara en la cual se aplica la funcion
    # y tambien la funcion a aplicar.
    prev = 0
    for l, a, c in params[:-1]:
        condlist.append(np.logical_and(dom < l, dom >= prev))
        funclist.append(lut(a, c))
        prev = l
    # Ultimo tramo (el limite superior esta definido implicitamente)
    (a_n, c_n) = params[-1]
    funclist.append(lut(a_n, c_n))

    return np.piecewise(dom, condlist, funclist)

def t_log(c):
    """
    Devuelve una transformación logarítmica con la forma:
    c * log(x+1)
    """

    dom = np.array(np.arange(256))
    return c*(np.log(dom+1)).clip(min=0.00001)

def t_pot(gamma, c):
    """
    Devuelve una transformación en potencia con la forma:\n
    c * (x^gamma)
    """

    dom = np.array(np.arange(256))
    return c*(dom**gamma)


##GRAFICAS
def plot(imgs, figsize=(10,10), patron=(1,1), gray=True, titulos=[], vmin=0, vmax=255):
    """
    Dibuja una serie de imagenes
    """

    assert patron[0]*patron[1] >= len(imgs), 'ERROR en el patrón del subplot.\n Verifica que el parámetro \'patron\' contemple la cantidad suficiente de imágenes'

    assert len(titulos)==0 or len(titulos)==len(imgs), 'ERROR la cantidad de imágenes debe ser igual a la cantidad de títulos'

    plt.figure(figsize=figsize)
    
    for i, img in enumerate(imgs):
        plt.subplot(patron[0], patron[1], i+1)
        if(len(titulos)>0): 
            plt.title(titulos[i])
        if(gray):
            plt.imshow(img.astype('uint8'), cmap='gray', vmin=vmin, vmax=vmax)
        else:
            plt.imshow(img.astype('uint8'), vmin=vmin, vmax=vmax)


#BIT MAP
def bitmap(img, profundidad=8):
    """
    Genera los bit maps de una imagen con valores de grises.
    @param img: imagen de entrada
    @param profundidad: cantidad de bits máximos que extraer
    @return: array con tantas imagenes como 'profundidad'
    """
    img_shape = img.shape

    assert len(img_shape)==2, "Utilizar una imagen en grises, o solo pasar un sólo canal."

    bit_maps = []

    for i in np.arange(profundidad):
        new_map = np.zeros((img_shape[0], img_shape[1]))
        for x in np.arange(img_shape[0]):
            for y in np.arange(img_shape[1]):
                new_map[x, y] = np.bitwise_and(img[x,y], 2**i)
        
        bit_maps.append(new_map)
        
    return bit_maps


#VARIAS
def get_y_values(x1,y1,x2,y2):
    """
    Devuelve los valores de la imagen de un segmento que va desde (x1,y1) a (x2,y2)
    """
    assert x2 > x1, 'ERROR, x2 debe ser mayor que x1'
    
    m = (y2-y1)/(x2-x1)
    perfil = np.zeros(x2-x1,dtype=int)
    for i, x in enumerate(np.arange(x1,x2)):
        perfil[i] = int(y1 + m*(x2-x1))
        
    return perfil


def mse(imgA, imgB):
    """
    Calcula el MSE (Mean-Squared Error) entre dos imágenes de igual tamaño
    """

    assert imgA.shape == imgB.shape, f"ERROR, las imágenes deben ser del mismo tamaño {imgA.shape} vs {imgB.shape}"

    err = np.sum((imgA.astype("float") - imgB.astype("float"))**2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err

def padding(img, pad=1, pad_number=0):
    """
    Agrega un borde de \'pad\' píxeles de ancho, con el valor dado por \'pad_number\'
    Por defecto se hace un padding con '\0\'

    @param pad: ancho del padding
    @param pad_number: valor con el que se hará el padding
    """
    assert pad > 0, f"ERROR, el valor de \'pad\' debe ser mayor a {0}"

    new_img = np.ones((img.shape[0]+2*pad, img.shape[1]+2*pad), dtype='uint8')*pad_number
    new_img[pad:-pad, pad:-pad] = img
    
    return new_img