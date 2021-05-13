import numpy as np
import cv2 as cv

def about():
    print('Funciones referidas al filtrado espacial')


def get_kernel_prom(size):
    """
    Genera un kernel de tamaño size X size, donde todos los elementos valen lo mismo
    """

    assert size > 0, 'ERROR, el tamaño del kernel no puede ser menor que 0'

    return np.ones((size,size))/(size**2)


def get_kernel_cruz(size):
    """
    Genera un kernel cruz de tamaño size X size donde sólo la cruz principal tiene valores != 0
    """
    
    assert size > 0, 'ERROR, el tamaño del kernel no puede ser menor que 0'

    kernel = np.zeros((size, size))
    mitad = int(size/2)
    valor_celda = 1/(size*2-1)
    kernel[:,mitad] = valor_celda
    kernel[mitad,:] = valor_celda
    return kernel


def get_kernel_suma_1(size):
    """
    Genera un kernel de suma 1
    """
    
    assert size > 0, 'ERROR, el tamaño del kernel no puede ser menor que 0'

    kernel = np.zeros((size,size))
    kernel[:,int(size/2)] = -1
    kernel[int(size/2),:] = -1
    kernel[int(size/2),int(size/2)] = size*2-1
    return kernel


def get_kernel_suma_0(size):
    """
    Genera un kernel de suma 0
    """
    
    assert size > 0, 'ERROR, el tamaño del kernel no puede ser menor que 0'

    kernel = np.ones((size,size))*-1
    kernel[int(size/2),int(size/2)] = size*size-1
    return kernel


def aplicar_mascara_difusa(img, kernel):
    """
    Aplica un filtro de máscara difusa a una imagen.
    """
    return img - cv.filter2D(img, -1, kernel)


def aplicar_alta_potencia(img, a, kernel):
    """
    Aplica un filtro de alta potencia a una imagen.
    
    @param a: aporte de la imagen original
    """
    return a*img - cv.filter2D(img, -1, kernel)