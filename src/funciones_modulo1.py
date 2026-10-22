import cv2
import matplotlib.pyplot as plt
import numpy as np
#Clase para manejar má facilmente la apertura y cierre del video y la obtencion de frames
class VideoReader:
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = None

    # Uso context manager para abrir con "with"
    def __enter__(self):
        self.video = cv2.VideoCapture(str(self.video_path))
        if not self.video.isOpened():
            self.video.release()
            self.video = None
            raise ValueError(f"No se pudo abrir el video: {self.video_path}") #Si el video no se puede abrir, tira error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.video is not None:
            self.video.release()
            self.video = None
        return False   #no suprime las excepciones

    # Propiedades
    @property
    def total_frames(self): #total de frames
        video = self._get_video()
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self): #frames por segundo
        video = self._get_video()
        return video.get(cv2.CAP_PROP_FPS)

    # Métodos
    def get_frame(self, frame_num):
        #Devuelve el frame número frame_num o None si no se pudo leer
        video = self._get_video()
        if not 0 <= frame_num < self.total_frames:
            raise IndexError(
                f"frame_num fuera de rango (0 a {self.total_frames - 1})"
            )
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if not ret:
            raise ValueError(f"No se pudo extraer el frame {frame_num}")
        return frame

    def get_frame_at_second(self, segundo):
        #Devuelve el frame correspondiente a un instante en segundos
        frame_num = int(segundo * self.fps)
        return self.get_frame(frame_num)

    # Método auxiliar para obtener el video
    def _get_video(self):
        if self.video is None:
            raise RuntimeError(
                "El video no está abierto. Usá la clase dentro de un bloque 'with'."
            )
        return self.video

#FUNCIONES DE PROCESAMIENTO
# 1.1.4 Cuantizacion (profundidad de bits)
 
# Funcion para reducir la cantidad de niveles de intensidad
def cuantizar(imagen, bits):
    niveles = 2 ** bits
    paso = 256 // niveles
 
    imagen_cuantizada = (imagen // paso) * paso
 
    return imagen_cuantizada.astype("uint8")
 
 
# 1.2.2 Escalado por rango de interes
 
def escalar_por_rango(roi_gris, p_bajo=5, p_alto=95):
    # Definir el intervalo de intensidades de interes
    f1 = np.percentile(roi_gris, p_bajo)
    f2 = np.percentile(roi_gris, p_alto)
    #Para entender: f1 y f2 representan los límites inferior y superior del intervalo 
    # que se desea expandir al rango completo de 0 a 255. Agarramos el 5% y el 95% de 
    # de las intensidades de la ROI en escala de grises. De esta manera, el intervalo
    # seleccionado contiene aproximadamente el 90% central de los valores de intensidad, 
    # excluyendo los extremos de la distribución.
 
    # Aplicar transformacion lineal: las intensidades entre f1 y f2 se expanden al rango 0-255 
    #Los valores inferiores a f1 se les asigna 0 y los valores mayores a f2 se les asigna 255
    roi_escalada = (roi_gris.astype(float) - f1) * 255 / (f2 - f1)
 
    # Limitar el resultado al rango [0, 255]
    roi_escalada = np.clip(roi_escalada, 0, 255).astype("uint8")
 
    return roi_escalada, f1, f2
 
 
# 1.2.4 Analisis de ruido y filtrado espacial
# B. Simulacion de los distintos tipos de ruido
 
def agregar_ruido_gaussiano(roi_gris, rng, sigma=20):
    # Matriz de gaussiano
    ruido_gaussiano = rng.normal(0, sigma, roi_gris.shape)
 
    #Sumamos el ruido gaussiano a las intensidades originales
    roi_gaussiana = np.clip(
        roi_gris.astype(float) + ruido_gaussiano, 0, 255
    ).astype("uint8")
 
    return roi_gaussiana
 
 
def agregar_sal_pimienta(roi_gris, rng, probabilidad=0.05):
    # Ruido sal y pimienta
    roi_sal_pimienta = roi_gris.copy()
 
    #Seleccionamos aleatoriamente qué pixeles vamos a modificar para el ruido sal y pimienta
    #queremos modificar el 5% de los pixeles de la imagen
    aleatorios = rng.random(roi_gris.shape) #generamos numeros aleatorios
 
    roi_sal_pimienta[aleatorios < probabilidad / 2] = 0
    roi_sal_pimienta[(aleatorios >= probabilidad / 2) &
                     (aleatorios < probabilidad)] = 255
 
    return roi_sal_pimienta
 
 
def agregar_speckle(roi_gris, rng, sigma=0.2):
    # Ruido speckle
    ruido_speckle = rng.normal(0, sigma, roi_gris.shape)
    roi_speckle = np.clip(
        roi_gris.astype(float) * (1 + ruido_speckle), 0, 255
    ).astype("uint8")
 
    return roi_speckle
 
 
# 1.3.1 Deteccion de bordes
 
# Funcion para aplicar los cinco operadores
def detectar_bordes(imagen):
 
    imagen_float = imagen.astype(np.float32)
 
    # Roberts
    kernel_roberts_x = np.array([[1, 0],
                                 [0, -1]], dtype=np.float32)
 
    kernel_roberts_y = np.array([[0, 1],
                                 [-1, 0]], dtype=np.float32)
 
    roberts_x = cv2.filter2D(
        imagen_float, cv2.CV_32F, kernel_roberts_x
    )
    roberts_y = cv2.filter2D(
        imagen_float, cv2.CV_32F, kernel_roberts_y
    )
 
    roberts = cv2.magnitude(roberts_x, roberts_y)
 
    # Sobel
    sobel_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
 
    sobel = cv2.magnitude(sobel_x, sobel_y)
 
    # Prewitt
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)
 
    kernel_prewitt_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]], dtype=np.float32)
 
    prewitt_x = cv2.filter2D(
        imagen_float, cv2.CV_32F, kernel_prewitt_x
    )
    prewitt_y = cv2.filter2D(
        imagen_float, cv2.CV_32F, kernel_prewitt_y
    )
 
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
 
    # Laplaciano
    laplaciano = cv2.Laplacian(
        imagen, cv2.CV_32F, ksize=3
    )
    laplaciano = np.abs(laplaciano)
 
    # Canny
    canny = cv2.Canny(imagen, 50, 150)
 
    return [roberts, sobel, prewitt, laplaciano, canny]
 
 
# 1.3.2 Transformada de Fourier 2D
# D. Visualizacion comparativa
 
# Normalizar cada reconstruccion para visualizarla
def normalizar_imagen(imagen):
    return cv2.normalize(
        imagen.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX
    )
 
 
# 1.3.4 Filtrado en el dominio frecuencial
 
def filtrar_frecuencial(frame_gris, fraccion_corte=0.10):
    # Calcular la transformada de Fourier del frame completo
    F_frame = np.fft.fft2(frame_gris.astype(float))
 
    # Centrar las frecuencias bajas
    F_frame_centrada = np.fft.fftshift(F_frame)
 
    # Obtener las dimensiones del frame
    alto_frame, ancho_frame = frame_gris.shape
 
    # Crear coordenadas para calcular la distancia al centro
    Y, X = np.ogrid[:alto_frame, :ancho_frame]
 
    centro_y = alto_frame // 2
    centro_x = ancho_frame // 2
 
    distancia = np.sqrt(
        (X - centro_x)**2 + (Y - centro_y)**2
    )
 
    # Definir una frecuencia de corte inicial
    D0 = fraccion_corte * min(alto_frame, ancho_frame)
 
    # Crear las mascaras de filtrado
    # Pasa-bajos: conserva frecuencias cercanas al centro
    H_pasabajos = (distancia <= D0).astype(float)
 
    # Pasa-altos: conserva frecuencias alejadas del centro
    H_pasaaltos = (distancia > D0).astype(float)
 
    # Aplicar las mascaras en el dominio frecuencial
    F_pasabajos = F_frame_centrada * H_pasabajos
    F_pasaaltos = F_frame_centrada * H_pasaaltos
 
    # Reconstruir las imagenes mediante transformada inversa
    frame_pasabajos = np.fft.ifft2(
        np.fft.ifftshift(F_pasabajos)
    ).real
 
    frame_pasaaltos = np.fft.ifft2(
        np.fft.ifftshift(F_pasaaltos)
    ).real
 
    return frame_pasabajos, frame_pasaaltos, H_pasabajos, H_pasaaltos
#FUNCIONES DE VISUALIZACIÓN
#Función de graficar una imagen
def mostrar_figura(size ,titulo, imagen):
    plt.figure(size)
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

#Función para graficar varias imágenes en una grid
def mostrar_imagenes(imagenes, titulos, filas=1, figsize=None,
                     cmap="gray", vmin=None, vmax=None):
    
    n = len(imagenes)
    columnas = -(-n // filas) 
    if figsize is None:
        figsize = (4 * columnas, 4 * filas)

    fig, ax = plt.subplots(filas, columnas, figsize=figsize)
    ax = ax.ravel() if n > 1 else [ax]

    for a, img, titulo in zip(ax, imagenes, titulos):
        a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(titulo, fontsize=10)
        a.axis("off")
    for a in ax[n:]: 
        a.axis("off")

    plt.tight_layout()
    plt.show()

#Función para mostrar lo histogramas en escala de grises
def mostrar_histogramas(imagenes, titulos, log=False, figsize=None):

    n = len(imagenes)
    fig, ax = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    ax = [ax] if n == 1 else ax

    for a, img, titulo in zip(ax, imagenes, titulos):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        a.plot(hist)
        if log:
            a.set_yscale("log")
        a.set_title(titulo)
        a.set_xlabel("Intensidad")
        a.set_ylabel("Cantidad de pixeles")
        a.set_xlim([0, 255])

    plt.tight_layout()
    plt.show()
