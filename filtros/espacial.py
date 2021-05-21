import numpy as np
import cv2 as cv

def about():
    print('Funciones referidas al filtrado espacial')


def getkernel(size, celdas=[], centro=None, resto=None, cruz=None, esquinas=None):
    """
    Genera un kernel arbitrario.

    Es importante el orden de los parámetros.
    Siempre se modifican los valores en el siguiente orden de importancia:\n (el que esta arriba pisa el valor del de abajo)
    - centro
    - resto
    - cruz
    - esquinas

    Ejemplo: si yo pongo un valor para esquinas=1, y un valor para resto=0, las esquinas tendrán un valor de 0

    @param celdas: debe tener tantos elementos como el kernel necesite.\n
     Los valores se van poniendo por fila, de izquierda a derecha.
    @param centro: un valor específico para el centro del kernel.
    @param resto: un valor para todos los valores excepto el centro.
    @param cruz: un valor específico para la cruz (sin  contar el centro)
    @param esquinas: un valor para las esquinas del kernel.
    """
    kernel = np.zeros((size,size))
    mitad = int(size/2)

    if(len(celdas)>0):
        assert len(celdas)==size**2, f"ERROR, celdas debe ser una lista con {size**2} elementos"
        
        for ix, x in enumerate(np.arange(kernel.shape[0])):
            for iy, y in enumerate(np.arange(kernel.shape[1])):
                kernel[ix, iy] = celdas[ix*kernel.shape[0]+iy]
    else:
        if(esquinas!=None):
            kernel[0,0] = esquinas
            kernel[0,-1] = esquinas
            kernel[-1,0] = esquinas
            kernel[-1,-1] = esquinas
        if(cruz!=None):
            kernel[:,mitad] = cruz
            kernel[mitad,:] = cruz
        if(resto!=None):
            kernel[:,:] = resto
        if(centro!=None):
            kernel[mitad,mitad] = centro

    return kernel


def getkernel_prom(size):
    f"""
    Genera un kernel de tamaño \'size x size\', donde todos los elementos valen lo mismo
    """

    assert size > 0, 'ERROR, el tamaño del kernel debe ser mayor a 0'

    return np.ones((size,size))/(size**2)


def getkernel_cruz(size, centro=None, cruz=None):
    """
    Genera un kernel cruz de tamaño \'size X size\'.
    
    Por defecto hace el kernel con el mismo valor en toda la cruz.

    @param centro: valor del centro del kernel
    @param cruz: valor de la cruz del kernel
    """
    
    assert size > 0, 'ERROR, el tamaño del kernel no puede ser menor o igual a 0'
    
    kernel = np.zeros((size, size))
    mitad = int(size/2)

    if(centro!=None or cruz!=None):
        assert centro != None, "ERROR, se encesita un valor de \'centro\'"
        assert cruz != None, "ERROR, se necesita un valor de \'cruz\'"
        
        kernel[:,mitad] = cruz
        kernel[mitad,:] = cruz
        kernel[mitad, mitad] = centro

    else:
        valor_celda = 1/(size*2-1)
        kernel[:,mitad] = valor_celda
        kernel[mitad,:] = valor_celda

    return kernel


def aplicar_mascaradifusa(img, kernel):
    """
    Aplica un filtro de máscara difusa a una imagen.
    """
    return img - cv.filter2D(img, -1, kernel)


def aplicar_altapotencia(img, a, kernel):
    """
    Aplica un filtro de alta potencia a una imagen.
    
    @param a: aporte de la imagen original
    """
    return a*img - cv.filter2D(img, -1, kernel)