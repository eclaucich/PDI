import numpy as np
import filtros.espacial as espc
import cv2 as cv

def about():
    print('Funciones relacionadas a la segmentación en imágenes.')


def prewitt(img, umbral=0, dcruzadas=False):
    """
    Realiza la detección de bordes mediante Prewitt.
    
    @param img: imagen en escala de grises
    @param umbral: valor [0,255] para el binarizado del resultado. Si se deja en 0 no se hace binarización.
    @param dcruzadas: solo calcular las derivadas con respecto a x,y (False), o hacer las derivadas cruzadas (True)
    """
    umbral = np.clip(umbral, 0, 255)

    img_prewitt = np.zeros_like(img)
    
    prewitt_dx = espc.getkernel(3, celdas=[-1,-1,-1, 0,0,0, 1,1,1])
    img_dx = cv.filter2D(img, -1, prewitt_dx)
    img_prewitt += np.abs(img_dx)
    
    prewitt_dy = espc.getkernel(3, celdas=[-1,0,1, -1,0,1, -1,0,1])
    img_dy = cv.filter2D(img, -1, prewitt_dy)
    img_prewitt += np.abs(img_dy)
    
    if(dcruzadas):
        prewitt_dxy = espc.getkernel(3, celdas=[0,1,1, -1,0,1, -1,-1,0])
        img_dxy = cv.filter2D(img, -1, prewitt_dxy)
        img_prewitt += np.abs(img_dxy)
        
        prewitt_dyx = espc.getkernel(3, celdas=[-1,-1,0, -1,0,1, 0,1,1])
        img_dyx = cv.filter2D(img, -1, prewitt_dyx)
        img_prewitt += np.abs(img_dyx)
        
    if(umbral>0):
        img_prewitt[img_prewitt>=umbral] = 255
        img_prewitt[img_prewitt<umbral] = 0
        
    return img_prewitt


def sobel(img, dx, dy, ksize=3, ddepth=cv.CV_8U, scale=1, delta=0, borderType=cv.BORDER_DEFAULT):
    """
    Realiza la detección de bordes por Sobel.
    Se usa indicando el orden de las derivadas en x,y.
    El orden de las derivadas puede ser 0, pero no las dos al mismo tiempo.

    @param img: imagen en escalas de grises.
    @param dx: orden de la dervida en x
    @param dy: orden de la dervida en y
    @param ksize: tamaño de la vecindad, sólo puede ser -1,1,3,5 0 7. Si se pone -1 se usa Scharr.
    @param ddepth: definición de los datos (CV_8U, CV_64F)
    @param scale: se multiplica todo el resultado por un escalar.
    @param delta: se le suma un escalar a todo el resultado.
    @param bordertype: bordes a usar para las vecindades.
    """

    assert dx >= 0 and dy >= 0, f"ERROR, el orden de las derivadas no puede ser negativo o cero: dx={dx}, dy={dy}"
    assert ksize==-1 or ksize==1 or ksize==3 or ksize==5 or ksize==7, f"ERROR, el tamaño del kernel solo puede ser -1, 1, 3, 5 o 7. ksize={ksize}"
    
    if dx > 0:
        sobel_dx = cv.Sobel(img, ddepth, dx, 0, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
        sobel_dx = cv.convertScaleAbs(sobel_dx)
    
    if dy > 0:
        sobel_dy = cv.Sobel(img, ddepth, 0, dy, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
        sobel_dy = cv.convertScaleAbs(sobel_dy)
    
    if dx > 0 and dy > 0:
        sobel_d = cv.addWeighted(sobel_dx, 0.5, sobel_dy, 0.5, 0)
        return sobel_d
    elif dx>0:
        return sobel_dx
    else:
        return sobel_dy


def laplacian(img, ksize=3, ddepth=cv.CV_8U, scale=1, delta=0, borderType=cv.BORDER_DEFAULT):
    """ 
    Realiza la detección de bordes mediante el laplaciano.

    @param img: imagen en escalas de grises.
    @param ksize: tamaño de la vecindad, sólo puede ser 1,3,5 0 7.
    @param ddepth: definición de los datos (CV_8U, CV_64F)
    @param scale: se multiplica todo el resultado por un escalar.
    @param delta: se le suma un escalar a todo el resultado.
    @param bordertype: bordes a usar para las vecindades.
    """

    assert ksize==-1 or ksize==1 or ksize==3 or ksize==5 or ksize==7, f"ERROR, el tamaño del kernel solo puede ser 1,3,5 o 7. ksize={ksize}"
    
    lap = cv.Laplacian(img, ksize=ksize, ddepth=ddepth, scale=scale, delta=delta, borderType=borderType)
    
    return lap


def canny(img, lim_inf, lim_sup, ksize=3, magnitudPrecisa=False):
    """
    Realiza la detección de bordes por Canny.

    Todos los pixeles que estén por encima de lim_sup serán considerados bordes.
    Todos los píxeles que estén entre lim_inf y lim_sup serán bordes sólo si estan conectados con un borde que esté por encima de lim_sup.
    Todos los píxeles que estén por debajo de lim_inf son descartados. 

    @param img: imagen en escala de grises.
    @param lim_inf: límite inferior para la histéresis
    @param lim_sup: límite superior para la histéresis
    @param ksize: tamaño de la vecindad para la detección de bordes.
    @param magnitudPrecisa: calcular la magnitud del gradiente de manera aproximada: |Gx|+|Gy| (False), o de manera más precisa: (|Gx|^2+|Gy|^2)^1/2 (True)
    """
    return cv.Canny(img, lim_inf, lim_sup, apertureSize=ksize, L2gradient=magnitudPrecisa)


def hough(img, rho, theta, umbral):
    """
    Aplica la transformada de Hough a una imagen de bordes.
    Tener en cuenta que se recibe una imagen donde ya se hayan detectado los bordes (Canny o Sobel)

    @param img: imagen en escalas de grises
    @param rho: definición del eje \'rho\' (cuantización del eje)
    @param theta: definición del eje \'theta\' (cuantización del eje)
    @param umbral: umbral para el contador de la transformada (mientras más grande, más puntos deben caer sobre una recta)
    @return: Devuelve una lista con todas las líneas encontradas que superan el umbral, de la forma [[(rho0,theta0)], [(rho1, theta1)], ..., [(rhoN, thetaN)]]
    Ejemplo:
        - Si rho=1 => la discretización se hace de a un pixel
        - Si rho=10 => se hacen pasos de a 10 en la discretización (todos los que caigan entre 0 y 10, son contados para la misma región)
        - Lo mismo para theta.
        - Se pueden discretizar de manera distinta, podemos tener un rho muy definido y un theta mas cuantizado.

        Si la salida son dos lineas, va a tener una forma (2, 1, 2)
        Para pasarlo a coordenadas cartesianas:
            theta = linea[0,0,1]
            rho = linea[0,0,0]
            a = np.cos(theta)
            b = np.sin(theta)
            x = a*rho
            y = b*theta
    """
    return cv.HoughLines(img, rho, theta, umbral)


#Utiliza para el calucla de regiones por crecimiento
def _flooding(img, semilla, visitados, res, lim_inf, lim_sup):
    
    if(visitados[x,y]==1):
        return res

    x = semilla[0]
    y = semilla[1]
    
    visitados[x,y] = 1
    
    vecinos = []

    if(x-1 >= 0):
        vecinos.append((x-1,y))
    if(x+1 < img.shape[1]):
        vecinos.append((x+1,y))
    if(y-1 >= 0):
        vecinos.append((x,y-1))
    if(y+1 < img.shape[0]):
        vecinos.append((x,y+1))
    if(x-1 >= 0 and y-1>=0):
        vecinos.append((x-1,y-1))
    if(x-1 >= 0 and y+1<img.shape[0]):
        vecinos.append((x-1,y+1))
    if(x+1 < img.shape[1] and y-1>=0):
        vecinos.append((x+1,y-1))
    if(x+1 < img.shape[1] and y+1<img.shape[0]):
        vecinos.append((x+1,y+1))
      
    for v in vecinos:    
        if(img[v]>=lim_inf and img[v]<=lim_sup):
            res[v[0],v[1]] = 255
            res = _flooding(img, v, visitados, res, lim_inf, lim_sup)
        
    return res


def floodind(img, semilla, lims):
    """
    Segmentación mediante crecimiento de regiones.
    El criterio a cumplir para la región es el gradiente entre el pixel semilla y el vecino.
    Usa recursión, así que tenerle paciencia, para imágenes grandes puede demorar.
    
    @param img: imágenes en escalas de grises.
    @param semilla: coordenada (x,y) del pixel en el que empezar a crecer la región.
    @param lims: Desviación del valor de gris permitido, con respecto a la semilla, para formar parte de la región, de la forma (limite inferior, limite superior).
    @return: Devuelve una imagen con 255 en la región detectada y 0 por fuera.
    """

    visitados = np.zeros_like(img)
    res = np.zeros_like(img)

    lim_inf = img[semilla]-np.abs(lims[0])
    lim_sup = img[semilla]+np.abs(lims[1])

    res = _flooding(img, semilla, visitados, res, lim_inf, lim_sup)
    
    return res