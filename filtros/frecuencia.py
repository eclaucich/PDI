import numpy as np
import cv2 as cv
from skimage import draw
import warnings

def about():
    print('Funciones referidas al filtrado frecuencial')

#Calcula la transformada y la desplaza. Es necesario hacerlo en varios lados
def _get_fshift(img, mode='opencv'):

    assert mode=='numpy' or mode=='opencv', 'ERROR, parámetro \'mode\' incorrecto'

    if(mode=='numpy'):
        f = np.fft.fft2(img)
    else:
        f = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    return np.fft.fftshift(f)


def get_espectro(img):
    """
    Calcula el espectro en frecuencia de una imagen

    @param mode: bilbioteca con la que se calcula la transformada de Fourier. (numpy/opencv)
    """

    return 20*np.log(np.abs(_get_fshift(img, mode='numpy')).clip(0.00001))


def get_modulo_fase(img):
    """
    Calcula el modulo y fase de Fourier de una imagen

    @return (modulo, fase)
    """
    fshift = _get_fshift(img, mode='opencv')

    return cv.cartToPolar(fshift[:,:,0], fshift[:,:,1])



def eliminar_modulo(img):
    """
    Genera una imagen que solo mantiene la fase original y contiene un módulo unitario
    """
    dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)

    magn, phase = cv.cartToPolar(dft[:,:,0], dft[:,:,1])
    
    magn_1 = np.ones_like(magn)
    x, y = cv.polarToCart(magn_1, phase)
    
    dft[:,:,0] = x
    dft[:,:,1] = y

    dft = cv.idft(dft)
    return cv.magnitude(dft[:,:,0], dft[:,:,1])    


#TODO: Funcion que elimina la fase de una imagen
def eliminar_fase(img):
    pass


def nueva_modulo_fase(img, mod, fas):
    """
    Genera una nueva imagen a partir de \'img\' cambiándole su módulo y fase.

    @param img: imagen original
    @param mod: módulo que tendrá la nueva imagen
    @param fas: fase que tendrá la nueva imagen
    """
    #Obtener transformada de img
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    x, y = cv.polarToCart(mod, fas)

    #Cambiar modulo y fase por los nuevos
    dft[:,:,0] = x
    dft[:,:,1] = y
    
    #Invertir transformada
    dft = np.fft.ifftshift(dft)
    dft = cv.idft(dft)
    #devolver magnitud de la imagen
    return cv.magnitude(dft[:,:,0], dft[:,:,1])

def aplicar_filtro(img, filtro):
    """
    Aplica un filtro en frecuencia.

    @param filtro: filtro con el mismo tamaño que la imagen
    """

    assert img.shape == filtro.shape, f'ERROR, la imagen y el filtro deben tener los mismos tamaños\n {img.shape} vs {filtro.shape}'

    fshift = _get_fshift(img, mode='numpy')

    f_filtrada = np.multiply(fshift, filtro)

    f_ishift = np.fft.ifftshift(f_filtrada)

    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    
    return img_back


def FPB_ideal(fc, img_shape):
    """
    Genera un filtro pasa bajos ideal.

    @param fc: frecuencia de corte [0, 255]
    @param image_shape: shape de la imagen
    """
    if(fc<0 or fc >255):
        warnings.warn('\'fc\' fuera de rango. Se hará clipping a [0, 255]')

    FPB = np.zeros(img_shape)
    
    y = int(img_shape[0]/2)
    x = int(img_shape[1]/2)
    
    coords = draw.disk((y, x), fc, shape=img_shape)
    
    FPB[coords] = 1
    
    return FPB


def FPB_butt(fc, n, img_shape):
    """
    Genera un filtro pasa bajos Butterworth.

    @param fc: frecuencia de corte [0, 255]
    @param n: orden del filtro
    @param image_shape: shape de la imagen
    """

    assert n>=0, 'El orden del filtro no puede ser negativo'

    if(fc<0 or fc >255):
        warnings.warn('\'fc\' fuera de rango. Se hará clipping a [0, 255]')

    filas = np.zeros((1, img_shape[0])); filas[0,:] = np.arange(img_shape[0])
    colum = np.zeros((img_shape[1], 1)); colum[:,0] = np.arange(img_shape[1])
    
    half_f = int(img_shape[0]/2)
    half_c = int(img_shape[1]/2)
    
    D = np.sqrt((filas-half_f)**2 + (colum-half_c)**2)
    FPB = 1 / (1 + (D/fc)**(2*n))
    
    return FPB.T

#TODO: FPA_gauss -> que genera un filtro pasa bajos gaussiano

def FPA_ideal(fc, img_shape):
    """
    Genera un filtro pasa altos ideal.

    @param fc: frecuencia de corte [0, 255]
    @param image_shape: shape de la imagen
    """
    if(fc<0 or fc >255):
        warnings.warn('\'fc\' fuera de rango. Se hará clipping a [0, 255]')

    FPA = np.ones(img_shape)
    
    y = int(img_shape[0]/2)
    x = int(img_shape[1]/2)
    
    coords = draw.disk((y, x), fc, shape=img_shape)
    
    FPA[coords] = 0
    
    return FPA


def FPA_butt(fc, n, image_shape):
    """
    Genera un filtro pasa altos Butterworth.

    @param fc: frecuencia de corte [0, 255]
    @param n: orden del filtro
    @param image_shape: shape de la imagen
    """

    assert n>=0, 'El orden del filtro no puede ser negativo'

    if(fc<0 or fc >255):
        warnings.warn('\'fc\' fuera de rango. Se hará clipping a [0, 255]')

    filas = np.zeros((1, image_shape[0])); filas[0,:] = np.arange(image_shape[0])
    colum = np.zeros((image_shape[1], 1)); colum[:,0] = np.arange(image_shape[1])
    
    half_f = int(image_shape[0]/2)
    half_c = int(image_shape[1]/2)
    
    D = np.sqrt((filas-half_f)**2 + (colum-half_c)**2).clip(min=0.00001)
    FPA = 1 / (1 + (fc/D)**(2*n))
    
    return FPA.T


def FPA_alta_potencia(a, fpa):
    """
    Genera un filtro de alta potencia

    @param a: aporte de la continua
    @param fpa: un filtro pasa altos
    @return: la salida será del mismo tamaño que \'fpa\'
    """
    return (a-1) + fpa


def FPA_enfasis(a,b,fpa):
    """
    Genera un filtro de énfasis

    @param a: aporte de la continua
    @param b: aporte del filtro pasa altos
    @param fpa: un filtro pasa altos
    @return: la salida será del mismo tamaño que \'fpa\'
    """
    return a + b*fpa


def aplicar_homomorfico(img, low, high, fc, n):
    """
    Aplica un filtro homomórfico a la imagen

    @param low: imagen de salida mínimo
    @param high: imagen de salida máximo
    @param fc: frecuencia de corte en [0, 255]
    @param n: orden del filtro 
    @retun (filtro aplicado, imagen filtrada)
    """

    assert n>=0, 'El orden del filtro no puede ser negativo'

    if(fc<0 or fc >255):
        warnings.warn('\'fc\' fuera de rango. Se hará clipping a [0, 255]')
        fc = fc.clip(min=0, max=255)

    #LOG
    log_img = np.log(img.clip(0.00001))
    
    #DFT
    dft_img = np.fft.fft2(log_img)
    dft_img = np.fft.fftshift(dft_img)
    
    #FILTRO H
    image_shape = img.shape
    filas = np.zeros((1, image_shape[1])); filas[0,:] = np.arange(image_shape[1])
    colum = np.zeros((image_shape[0], 1)); colum[:,0] = np.arange(image_shape[0])
    
    half_f = int(image_shape[1]/2)
    half_c = int(image_shape[0]/2)
    
    D = np.sqrt((filas-half_f)**2 + (colum-half_c)**2).clip(min=0.0001)
    H = low + ((high-low) / (1 + (fc/D)**(2*n)))
    
    h_img = np.multiply(H, dft_img)
    
    #IDFT
    idft_img = np.fft.ifftshift(h_img)
    idft_img = np.fft.ifft2(idft_img)
    
    #EXP
    return (np.abs(np.exp(np.real(idft_img))), H)