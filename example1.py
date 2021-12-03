import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob


def DFFTnp(img):
    f = np.fft.fft2(img) #двумерное дискретное преобразование Фурье
    fshift = np.fft.fftshift(f) #сдвиг компонента нулевой частоты в центр спектра
    #возвращает сдвинутый массив
    return fshift


def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft) #сдвиг компонента нулевой частоты в центр спектра
    reverse_image = np.fft.ifft2(f_ishift) #двумерное дискретное преобразование Фурье
    # возвращает сдвинутый массив
    return reverse_image


def showDFFT(img, fft, name):
    magnitude = np.abs(fft) #модуль для всех элементов массива
    plt.subplot(121), plt.imshow(img, 'Greys', vmin=0, vmax=255) #subplot- облать для графиков/фото, imshow-показ изображения
    # plt.title('Input Image ' + name), plt.xticks([]), plt.yticks([])  # заголовок

    s_min = magnitude.min()
    s_max = magnitude.max()
    if s_min == s_max:
        plt.subplot(122), plt.imshow(magnitude, 'Greys', vmin = 0, vmax = 255) #выводим с особыми ограничениями изображение
    else:
        plt.subplot(122), plt.imshow(magnitude, 'Greys')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()


def SOBEL(name):
    img = np.float32(cv.imread(name+'.png', 0))
    fshift = DFFTnp(img)
    # SOBEL  r"E:\1\cat.jpg"
    ksize = 3
    kernel = np.zeros(img.shape)
    sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel[0:ksize, 0:ksize] = sobel_h
    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)
    reverse_image = reverseDFFTnp(mult)
    plt.imshow(abs(reverse_image), cmap='gray')
    plt.title("Sobel")
    plt.show()

def BLUR(name):
    img = np.float32(cv.imread(name+'.png', 0))
    fshift = DFFTnp(img)
    ksize = 21
    kernel = np.zeros(img.shape)
    blur = cv.getGaussianKernel(ksize, -1)
    blur = np.matmul(blur, np.transpose(blur))
    kernel [0:ksize, 0:ksize ] = blur
    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)
    reverse_image = reverseDFFTnp(mult)
    plt.imshow(abs(reverse_image), cmap='gray')
    plt.title("Gauss blur")
    plt.show()

def CRAZY(name):
        img = np.float32(cv.imread(name + '.png', 0))
        fshift = DFFTnp(img)
        ksize = 3
        kernel = np.zeros(img.shape)
        sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel[0:ksize, 0:ksize] = sobel_h
        fkshift = DFFTnp(kernel)
        mult = np.multiply(fshift, fkshift)
        reverse_image = reverseDFFTnp(mult)

        ksize = 21
        kernel1 = np.zeros(reverse_image.shape)
        blur = cv.getGaussianKernel(ksize, -1)
        blur = np.matmul(blur, np.transpose(blur))
        kernel1[0:ksize, 0:ksize] = blur
        fkshift1 = DFFTnp(kernel1)
        mult1 = np.multiply(fkshift1, mult)
        reverse_image1 = reverseDFFTnp(mult1)

        plt.imshow(abs(reverse_image1), cmap='gray')
        plt.title("CRAZY")
        plt.show()



folder_path = r"C:\Users\diana\OneDrive\Desktop\stripes\__129"

images = glob.glob(folder_path + '*.png')
for name in images:
    img = np.float32(cv.imread(name, 0))
    f = np.fft.fft2(img)  #двумерное дискретное преобразование Фурье
    fshift = np.fft.fftshift(f) #сдвиг компонента нулевой частоты в центр спектра
    #showDFFT(img, fshift, name)
    SOBEL(folder_path)
    BLUR(folder_path)
    CRAZY(folder_path)