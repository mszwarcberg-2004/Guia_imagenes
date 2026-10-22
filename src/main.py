#Imports
import cv2 #cv2: librería que permite abrir videos, extraer frames y procesar imagenes.
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#Módulo 1:  

# Ruta del video
CARPETA_PROYECTO =CARPETA_PROYECTO = Path.cwd()
VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "video1.mov"

# Abrir el video
video = cv2.VideoCapture(str(VIDEO_PATH))

#Si el video no se puede abrir, tira error
if not video.isOpened():
    raise ValueError("No se pudo abrir el <video")

#1.1.1

# Obtener la cantidad de frames
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


# Seleccionar un frame en un instante determinado
segundo = 6

fps = video.get(cv2.CAP_PROP_FPS)
frame_num = int(segundo * fps)

video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

# Extraer el frame
ret, frame = video.read()
video.release()

if not ret:
    raise ValueError("No se pudo extraer el frame")

# Convertir de BGR (en lo que trabaja opencvs) a RGB para visualizarlo
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mostrar el frame
plt.figure(figsize=(8, 6))
plt.imshow(frame_rgb)
plt.title("Frame representativo del video")
plt.axis("off")
plt.show()

# Ruta donde se guarda la ROI
ROI_PATH = CARPETA_PROYECTO / "imagenes" / "roi.png"

# Verificar si ya existe una ROI guardada
if ROI_PATH.exists():

    # Si existe, cargarla directamente
    roi = cv2.imread(str(ROI_PATH))
    print("Se cargo la ROI guardada")

else:

    # Si no existe, seleccionar una nueva ROI
    x, y, w, h = cv2.selectROI("Seleccionar ROI", frame, False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise ValueError("No se selecciono una ROI")

    roi = frame[y:y+h, x:x+w]

    # Guardar la ROI seleccionada como png en mi compu
    cv2.imwrite(str(ROI_PATH), roi)

# Crear carpeta IMAGENES, donde vamos a guardar el frame y el roi
CARPETA_IMAGENES = CARPETA_PROYECTO / "imagenes"
CARPETA_IMAGENES.mkdir(exist_ok=True)

# Guardar frame completo como png en mi compu
cv2.imwrite(str(CARPETA_IMAGENES / "frame.png"), frame)


# Visualizar ROI
roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(roi_rgb)
plt.title("Region de interes (ROI)")
plt.axis("off")
plt.show()


# 1.1.2 Dimensiones y profundidad de bits

alto, ancho, canales = roi.shape

# Cantidad de bits por canal
bits = roi.dtype.itemsize * 8

print("Dimensiones de la ROI:")
print(f"Ancho: {ancho} pixeles")
print(f"Alto: {alto} pixeles")
print(f"Cantidad de canales: {canales}")
print(f"Profundidad de bits por canal: {bits}")
print(f"Profundidad total por pixel: {bits * canales} bits")


# 1.1.3 Resolucion espacial: submuestreo

escalas = [0.50, 0.25, 0.05]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(roi_rgb)
plt.title("ROI original (100%)")
plt.axis("off")

for i, escala in enumerate(escalas):
    nuevo_ancho = max(1, round(ancho * escala))
    nuevo_alto = max(1, round(alto * escala))

    roi_reducida = cv2.resize(
        roi_rgb,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )

    plt.subplot(2, 2, i + 2)
    plt.imshow(roi_reducida, interpolation="nearest")
    plt.title(f"Resolucion {int(escala * 100)}% ({nuevo_ancho}x{nuevo_alto})")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 1.1.4 Cuantizacion (profundidad de bits)

# Funcion para reducir la cantidad de niveles de intensidad
def cuantizar(imagen, bits):
    niveles = 2 ** bits
    paso = 256 // niveles

    imagen_cuantizada = (imagen // paso) * paso

    return imagen_cuantizada.astype("uint8")


# Cuantizacion de la ROI a 4 y 2 bits por canal
roi_4bits = cuantizar(roi_rgb, 4)
roi_2bits = cuantizar(roi_rgb, 2)

# Visualizacion comparativa
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(roi_rgb)
plt.title("Original - 8 bits")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(roi_4bits)
plt.title("Cuantizacion - 4 bits")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(roi_2bits)
plt.title("Cuantizacion - 2 bits")
plt.axis("off")

plt.tight_layout()
plt.show()


# 1.1.5 Espacios de color

# A. Separacion de los canales RGB

# Separar la ROI en sus tres canales de color
R, G, B = cv2.split(roi_rgb)

# Visualizar cada canal en escala de grises
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap="gray", vmin=0, vmax=255)
plt.title("Canal R (Rojo)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(G, cmap="gray", vmin=0, vmax=255)
plt.title("Canal G (Verde)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(B, cmap="gray", vmin=0, vmax=255)
plt.title("Canal B (Azul)")
plt.axis("off")

plt.tight_layout()
plt.show()


# B. Conversion del espacio RGB a HSV

# Convertir la ROI de RGB a HSV
roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

# Separar las componentes HSV
H, S, V = cv2.split(roi_hsv)

# Visualizar cada componente en escala de grises
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(H, cmap="gray", vmin=0, vmax=179)
plt.title("H - Matiz")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(S, cmap="gray", vmin=0, vmax=255)
plt.title("S - Saturacion")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(V, cmap="gray", vmin=0, vmax=255)
plt.title("V - Valor")
plt.axis("off")

plt.tight_layout()
plt.show()

#1.2.1 

# Convertir el frame de BGR a escala de grises
frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Aumentar el brillo sumando 50 a cada pixel
# np.clip limita las intensidades al rango [0, 255]
frame_brillo = np.clip(frame_gris.astype(float) + 50, 0, 255).astype("uint8")

# Aumentar el contraste multiplicando por 1.5
frame_contraste = np.clip(frame_gris.astype(float) * 1.5, 0, 255).astype("uint8")

# Visualizar las imagenes
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(frame_gris, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Frame original")

ax[1].imshow(frame_brillo, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Brillo +50")

ax[2].imshow(frame_contraste, cmap="gray", vmin=0, vmax=255)
ax[2].set_title("Contraste x1.5")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

# Calcular y comparar los histogramas
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

imagenes = [frame_gris, frame_brillo, frame_contraste]
titulos = ["Histograma original", "Brillo +50", "Contraste x1.5"]

for i in range(3):
    histograma = cv2.calcHist([imagenes[i]], [0], None, [256], [0, 256])
    ax[i].plot(histograma)
    ax[i].set_yscale("log") #cambiamos la escala del eje y a logarítmica
    ax[i].set_title(titulos[i])
    ax[i].set_xlabel("Intensidad")
    ax[i].set_ylabel("Cantidad de pixeles")
    ax[i].set_xlim([0, 255])

plt.tight_layout()
plt.show()


# 1.2.2 Escalado por rango de interes
# Convertir la ROI a escala de grises
roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Definir el intervalo de intensidades de interes
f1 = np.percentile(roi_gris, 5)
f2 = np.percentile(roi_gris, 95)
#Para entender: f1 y f2 representan los límites inferior y superior del intervalo 
# que se desea expandir al rango completo de 0 a 255. Agarramos el 5% y el 95% de 
# de las intensidades de la ROI en escala de grises. De esta manera, el intervalo
# seleccionado contiene aproximadamente el 90% central de los valores de intensidad, 
# excluyendo los extremos de la distribución.

print(f"f1 = {f1:.1f}")
print(f"f2 = {f2:.1f}")

# Aplicar transformacion lineal: las intensidades entre f1 y f2 se expanden al rango 0-255 
#Los valores inferiores a f1 se les asigna 0 y los valores mayores a f2 se les asigna 255
roi_escalada = (roi_gris.astype(float) - f1) * 255 / (f2 - f1)

# Limitar el resultado al rango [0, 255]
roi_escalada = np.clip(roi_escalada, 0, 255).astype("uint8")

# Visualizar las imagenes
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(roi_gris, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("ROI original")

ax[1].imshow(roi_escalada, cmap="gray", vmin=0, vmax=255)
ax[1].set_title(f"Escalado lineal [{f1:.1f}, {f2:.1f}]")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

# Comparar los histogramas
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

hist_original = cv2.calcHist([roi_gris], [0], None, [256], [0, 256])
hist_escalado = cv2.calcHist([roi_escalada], [0], None, [256], [0, 256])

ax[0].plot(hist_original)
ax[0].set_title("Histograma original")

ax[1].plot(hist_escalado)
ax[1].set_title("Histograma escalado")

for a in ax:
    a.set_xlabel("Intensidad")
    a.set_ylabel("Cantidad de pixeles")
    a.set_xlim([0, 255])

plt.tight_layout()
plt.show()

# 1.2.3 Ecualización del histograma

# Ecualizacion global
roi_ecualizada = cv2.equalizeHist(roi_gris)

# Ecualizacion local (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
roi_clahe = clahe.apply(roi_gris)

# Comparar las imagenes
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

imagenes = [roi_gris, roi_escalada, roi_ecualizada, roi_clahe]
titulos = ["ROI original", "Escalado lineal", "Ecualizacion global", "CLAHE"]

for i in range(4):
    ax[i].imshow(imagenes[i], cmap="gray", vmin=0, vmax=255)
    ax[i].set_title(titulos[i])
    ax[i].axis("off")

plt.tight_layout()
plt.show()

# Comparar los histogramas
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    histograma = cv2.calcHist([imagenes[i]], [0], None, [256], [0, 256])

    ax[i].plot(histograma)
    ax[i].set_title(titulos[i])
    ax[i].set_xlabel("Intensidad")
    ax[i].set_ylabel("Cantidad de pixeles")
    ax[i].set_xlim([0, 255])

plt.tight_layout()
plt.show()


# 1.2.4 Analisis de ruido y filtrado espacial

# A. Aplicacion de filtros sobre la ROI original (sin ruido simulado)

# Filtro de media
roi_media = cv2.blur(roi_gris, (3, 3))

# Filtro de mediana
roi_mediana = cv2.medianBlur(roi_gris, 3)

# Filtro gaussiano
roi_filtro_gaussiano = cv2.GaussianBlur(roi_gris, (3, 3), 0)

# Comparar la imagen original con las tres filtradas
imagenes = [roi_gris, roi_media, roi_mediana, roi_filtro_gaussiano]
titulos = ["ROI original", "Filtro de media", "Filtro de mediana", "Filtro gaussiano"]

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    ax[i].imshow(imagenes[i], cmap="gray", vmin=0, vmax=255)
    ax[i].set_title(titulos[i])
    ax[i].axis("off")

plt.tight_layout()
plt.show()


# B. Simulacion de los distintos tipos de ruido

# Creamos un generador de numeros aleatorios
rng = np.random.default_rng(42)

# Matriz de gaussiano
ruido_gaussiano = rng.normal(0, 20, roi_gris.shape)

#Sumamos el ruido gaussiano a las intensidades originales
roi_gaussiana = np.clip(
    roi_gris.astype(float) + ruido_gaussiano, 0, 255
).astype("uint8")

# Ruido sal y pimienta
roi_sal_pimienta = roi_gris.copy()

#Seleccionamos aleatoriamente qué pixeles vamos a modificar para el ruido sal y pimienta
probabilidad = 0.05 #queremos modificar el 5% de los pixeles de la imagen
aleatorios = rng.random(roi_gris.shape) #generamos numeros aleatorios

roi_sal_pimienta[aleatorios < probabilidad / 2] = 0
roi_sal_pimienta[(aleatorios >= probabilidad / 2) &
                 (aleatorios < probabilidad)] = 255

# Ruido speckle
ruido_speckle = rng.normal(0, 0.2, roi_gris.shape)
roi_speckle = np.clip(
    roi_gris.astype(float) * (1 + ruido_speckle), 0, 255
).astype("uint8")

# Comparacion de los ruidos
imagenes_ruido = [
    roi_gris,
    roi_gaussiana,
    roi_sal_pimienta,
    roi_speckle
]

titulos_ruido = [
    "ROI original",
    "Ruido gaussiano",
    "Sal y pimienta",
    "Ruido speckle"
]

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    ax[i].imshow(imagenes_ruido[i], cmap="gray", vmin=0, vmax=255)
    ax[i].set_title(titulos_ruido[i])
    ax[i].axis("off")

plt.tight_layout()
plt.show()


# C. Aplicacion de filtros espaciales

# Aplicar filtro gaussiano al ruido gaussiano
roi_gaussiana_filtrada = cv2.GaussianBlur(roi_gaussiana, (3, 3), 0)

# Aplicar filtro de mediana al ruido sal y pimienta
roi_sal_pimienta_filtrada = cv2.medianBlur(roi_sal_pimienta, 3)

# Aplicar filtro de media al ruido speckle
roi_speckle_filtrada = cv2.blur(roi_speckle, (3, 3))

# Comparar imagenes antes y despues del filtrado
imagenes = [
    roi_gaussiana,
    roi_gaussiana_filtrada,
    roi_sal_pimienta,
    roi_sal_pimienta_filtrada,
    roi_speckle,
    roi_speckle_filtrada
]

titulos = [
    "Ruido gaussiano simulado",
    "Filtro gaussiano",
    "Ruido sal y pimienta simulado",
    "Filtro de mediana",
    "Ruido speckle simulado",
    "Filtro de media"
]

fig, ax = plt.subplots(3, 2, figsize=(8, 6))

for i in range(3):
    for j in range(2):
        ax[i, j].imshow(imagenes[i*2+j], cmap="gray", vmin=0, vmax=255)
        ax[i, j].set_title(titulos[i*2+j])
        ax[i, j].axis("off")

plt.tight_layout()
plt.show()


# 1.3 Deteccion de bordes y dominio frecuencial

# 1.3.1 Deteccion de bordes

# Imagen original y filtrada en el punto 1.2.4
imagen_original = roi_gris
imagen_filtrada = roi_filtro_gaussiano

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


# Aplicar los operadores a ambas imagenes
bordes_original = detectar_bordes(imagen_original)
bordes_filtrada = detectar_bordes(imagen_filtrada)

nombres = ["Roberts", "Sobel", "Prewitt",
           "Laplaciano", "Canny"]

# Comparacion visual
fig, ax = plt.subplots(2, 6, figsize=(18, 7))

ax[0, 0].imshow(imagen_original, cmap="gray")
ax[0, 0].set_title("ROI original")

ax[1, 0].imshow(imagen_filtrada, cmap="gray")
ax[1, 0].set_title("ROI filtrada - Gaussiano")

for i in range(5):

    # Usamos la misma escala por operador
    # para comparar original y filtrada
    limite = max(
        np.max(bordes_original[i]),
        np.max(bordes_filtrada[i]),
        1
    )

    ax[0, i+1].imshow(
        bordes_original[i], cmap="gray",
        vmin=0, vmax=limite
    )
    ax[0, i+1].set_title(nombres[i] + " - Original")

    ax[1, i+1].imshow(
        bordes_filtrada[i], cmap="gray",
        vmin=0, vmax=limite
    )
    ax[1, i+1].set_title(nombres[i] + " - Filtrada")

for fila in ax:
    for a in fila:
        a.axis("off")

plt.tight_layout()
plt.show()


# 1.3.2 Transformada de Fourier 2D

# Calcular la transformada de Fourier de la ROI original
F_original = np.fft.fft2(roi_gris)

# Centrar las frecuencias bajas
F_original_centrada = np.fft.fftshift(F_original)

# Obtener magnitud y fase
magnitud_original = np.abs(F_original_centrada)
fase_original = np.angle(F_original_centrada)

# Representar la magnitud en escala logaritmica
magnitud_log = np.log1p(magnitud_original)

# Visualizar la imagen, magnitud y fase
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(roi_gris, cmap="gray")
ax[0].set_title("ROI original")

ax[1].imshow(magnitud_log, cmap="gray")
ax[1].set_title("Magnitud (log)")

ax[2].imshow(fase_original, cmap="gray",
             vmin=-np.pi, vmax=np.pi)
ax[2].set_title("Fase")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# A. Reconstruccion utilizando solo magnitud

# Conservar magnitud y establecer fase igual a cero
F_solo_magnitud = magnitud_original

recon_magnitud = np.fft.ifft2(
    np.fft.ifftshift(F_solo_magnitud)
).real


# B. Reconstruccion utilizando solo fase

# Conservar fase y establecer magnitud igual a uno
F_solo_fase = np.exp(1j * fase_original)

recon_fase = np.fft.ifft2(
    np.fft.ifftshift(F_solo_fase)
).real


# C. Cruzar magnitud y fase de dos imagenes

# Fourier de la ROI filtrada con gaussiano
F_gaussiana = np.fft.fftshift(
    np.fft.fft2(roi_filtro_gaussiano)
)

magnitud_gaussiana = np.abs(F_gaussiana)
fase_gaussiana = np.angle(F_gaussiana)

# Magnitud original + fase gaussiana
F_cruzada_1 = magnitud_original * np.exp(
    1j * fase_gaussiana
)

recon_cruzada_1 = np.fft.ifft2(
    np.fft.ifftshift(F_cruzada_1)
).real

# Magnitud gaussiana + fase original
F_cruzada_2 = magnitud_gaussiana * np.exp(
    1j * fase_original
)

recon_cruzada_2 = np.fft.ifft2(
    np.fft.ifftshift(F_cruzada_2)
).real


# D. Visualizacion comparativa

# Normalizar cada reconstruccion para visualizarla
def normalizar_imagen(imagen):
    return cv2.normalize(
        imagen.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX
    )

imagenes = [
    roi_gris,
    recon_magnitud,
    recon_fase,
    recon_cruzada_1,
    recon_cruzada_2
]

titulos = [
    "ROI original",
    "Solo magnitud",
    "Solo fase",
    "Magnitud original + fase gaussiana",
    "Magnitud gaussiana + fase original"
]

fig, ax = plt.subplots(1, 5, figsize=(18, 4))

for i in range(5):
    ax[i].imshow(
        normalizar_imagen(imagenes[i]),
        cmap="gray"
    )
    ax[i].set_title(titulos[i], fontsize=9)
    ax[i].axis("off")

plt.tight_layout()
plt.show()


# 1.3.3 Densidad espectral de potencia (PSD)

# Calcular la transformada de Fourier de la ROI original y filtrada
F_original_psd = np.fft.fftshift(
    np.fft.fft2(roi_gris.astype(float))
)

F_gaussiana_psd = np.fft.fftshift(
    np.fft.fft2(roi_filtro_gaussiano.astype(float))
)

# Calcular la PSD como el cuadrado de la magnitud
psd_original = np.abs(F_original_psd)**2
psd_gaussiana = np.abs(F_gaussiana_psd)**2

# Aplicar escala logaritmica para visualizar mejor
psd_original_log = np.log1p(psd_original)
psd_gaussiana_log = np.log1p(psd_gaussiana)

# Usar la misma escala para comparar las dos imagenes
escala_maxima = max(
    np.max(psd_original_log),
    np.max(psd_gaussiana_log)
)

# Visualizar las PSD
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(
    psd_original_log,
    cmap="gray",
    vmin=0,
    vmax=escala_maxima
)
ax[0].set_title("PSD - ROI original")

ax[1].imshow(
    psd_gaussiana_log,
    cmap="gray",
    vmin=0,
    vmax=escala_maxima
)
ax[1].set_title("PSD - ROI filtrada (Gaussiano)")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# 1.3.4 Filtrado en el dominio frecuencial

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
D0 = 0.10 * min(alto_frame, ancho_frame)

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

# Aplicar filtro gaussiano en el dominio espacial
# para comparar ambos tipos de filtrado
frame_gaussiano = cv2.GaussianBlur(
    frame_gris, (5, 5), 0
)

# Visualizar las imagenes
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].imshow(frame_gris, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Frame original")

ax[1].imshow(frame_pasabajos, cmap="gray",
             vmin=0, vmax=255)
ax[1].set_title("Pasa-bajos frecuencial")

# El pasa-altos puede contener valores negativos.
# Centramos la visualizacion en cero para observar bordes.
limite = max(np.max(np.abs(frame_pasaaltos)), 1)

ax[2].imshow(frame_pasaaltos, cmap="gray",
             vmin=-limite, vmax=limite)
ax[2].set_title("Pasa-altos frecuencial")

ax[3].imshow(frame_gaussiano, cmap="gray",
             vmin=0, vmax=255)
ax[3].set_title("Gaussiano espacial")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# Visualizar las mascaras frecuenciales
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(H_pasabajos, cmap="gray")
ax[0].set_title("Mascara pasa-bajos")

ax[1].imshow(H_pasaaltos, cmap="gray")
ax[1].set_title("Mascara pasa-altos")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

# Preguntas finales del 1.3 - PSNR y SSIM

from skimage.metrics import structural_similarity as ssim

# Imagen de referencia: ROI original
imagen_referencia = roi_gris

# Imagenes obtenidas con los filtros espaciales del 1.2.4
imagenes_filtradas = {
    "Media": roi_media,
    "Mediana": roi_mediana,
    "Gaussiano": roi_filtro_gaussiano
}

# Calcular PSNR y SSIM para cada filtro
print("\n--- Comparacion PSNR y SSIM ---")

for nombre, imagen in imagenes_filtradas.items():

    # PSNR: compara las diferencias de intensidad
    psnr_valor = cv2.PSNR(imagen_referencia, imagen)

    # SSIM: compara la similitud estructural
    ssim_valor = ssim(
        imagen_referencia,
        imagen,
        data_range=255
    )

    print(f"\nFiltro: {nombre}")
    print(f"PSNR: {psnr_valor:.2f} dB")
    print(f"SSIM: {ssim_valor:.4f}")


# MODULO 2: Segmentacion y morfologia matematica de la pupila

# 2.1.1 Umbralizacion automatica de Otsu

# Utilizamos la ROI original en escala de grises
imagen_otsu = roi_gris

# Calcular automaticamente el umbral mediante Otsu
umbral, mascara_otsu = cv2.threshold(
    imagen_otsu,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

print(f"Umbral calculado por Otsu: {umbral}")

# Visualizar la ROI y la mascara obtenida
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(imagen_otsu, cmap="gray")
ax[0].set_title("ROI original")

ax[1].imshow(mascara_otsu, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Segmentacion mediante Otsu")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# 2.1.2 Segmentacion mediante K-Means

# Trabajamos con la ROI original en escala de grises
imagen_kmeans = roi_gris

# Convertir la imagen en una lista de pixeles
pixeles = imagen_kmeans.reshape(-1, 1).astype(np.float32)

# Criterio de finalizacion del algoritmo
criterio = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

# Guardar las mascaras para compararlas despues
mascaras_kmeans = {}

fig, ax = plt.subplots(2, 4, figsize=(14, 7))

for i, k in enumerate([2, 3, 4, 5]):

    # Aplicar K-Means
    _, etiquetas, centros = cv2.kmeans(
        pixeles,
        k,
        None,
        criterio,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    # Reconstruir la imagen agrupada
    centros_uint8 = np.uint8(centros)

    imagen_agrupada = centros_uint8[
        etiquetas.flatten()
    ].reshape(imagen_kmeans.shape)

    # Identificar el grupo de menor intensidad
    grupo_oscuro = np.argmin(centros.flatten())

    # Crear mascara binaria
    mascara = (
        etiquetas.flatten() == grupo_oscuro
    ).reshape(imagen_kmeans.shape).astype(np.uint8) * 255

    mascaras_kmeans[k] = mascara

    # Visualizar imagen agrupada
    ax[0, i].imshow(
        imagen_agrupada, cmap="gray",
        vmin=0, vmax=255
    )
    ax[0, i].set_title(f"K-Means K={k}")

    # Visualizar mascara del grupo mas oscuro
    ax[1, i].imshow(
        mascara, cmap="gray",
        vmin=0, vmax=255
    )
    ax[1, i].set_title(f"Grupo mas oscuro K={k}")

    print(f"K={k}: centros = {np.sort(centros.flatten())}")

for fila in ax:
    for a in fila:
        a.axis("off")

plt.tight_layout()
plt.show()



# 2.1.3 Segmentacion mediante contornos activos (Snakes)

from skimage.segmentation import active_contour
from skimage.filters import gaussian

# Imagen de trabajo en escala de grises
imagen_snake = roi_gris.astype(float) / 255.0

# Suavizar para reducir pequenas variaciones
imagen_snake_suavizada = gaussian(
    imagen_snake, sigma=1
)

# Dimensiones de la ROI
alto, ancho = roi_gris.shape

# Definir contorno inicial circular
centro_x = ancho / 2
centro_y = alto / 2

radio_inicial = min(alto, ancho) * 0.35

# Crear puntos del circulo
angulo = np.linspace(0, 2*np.pi, 120, endpoint=False)

# Coordenadas en formato fila, columna
contorno_inicial = np.array([
    centro_y + radio_inicial * np.sin(angulo),
    centro_x + radio_inicial * np.cos(angulo)
]).T

# Aplicar contorno activo
contorno_final = active_contour(
    imagen_snake_suavizada,
    contorno_inicial,
    alpha=0.015,
    beta=10,
    gamma=0.001,
    w_line=0,
    w_edge=1,
    max_num_iter=2500,
    boundary_condition="periodic"
)

# Crear mascara binaria a partir del contorno final
mascara_snake = np.zeros_like(roi_gris, dtype=np.uint8)

# OpenCV utiliza coordenadas columna, fila
puntos_snake = np.round(
    contorno_final[:, ::-1]
).astype(np.int32)

cv2.fillPoly(
    mascara_snake,
    [puntos_snake],
    255
)

# Visualizar el contorno inicial y final
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(roi_gris, cmap="gray")
ax[0].plot(
    contorno_inicial[:, 1],
    contorno_inicial[:, 0],
    "--r",
    label="Contorno inicial"
)
ax[0].plot(
    contorno_final[:, 1],
    contorno_final[:, 0],
    "-g",
    label="Contorno final"
)
ax[0].set_title("Segmentacion con Snakes")
ax[0].legend()

ax[1].imshow(
    mascara_snake,
    cmap="gray",
    vmin=0, vmax=255
)
ax[1].set_title("Mascara obtenida con Snakes")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# 2.1.4 Segmentacion mediante Watershed

# Imagen de trabajo
imagen_ws = roi_gris.copy()

# Suavizado gaussiano para reducir variaciones locales
imagen_ws_filtrada = cv2.GaussianBlur(
    imagen_ws, (3, 3), 0
)

# Utilizar la mascara de Otsu para definir regiones iniciales
binaria = (mascara_otsu > 0).astype(np.uint8)

# Componentes conectadas para identificar la pupila
cantidad, etiquetas, estadisticas, centroides = (
    cv2.connectedComponentsWithStats(binaria, 8)
)

alto_ws, ancho_ws = imagen_ws.shape
centro_roi = np.array([ancho_ws / 2, alto_ws / 2])

# Elegir la componente mas cercana al centro de la ROI
componentes = list(range(1, cantidad))

componente_pupila = min(
    componentes,
    key=lambda i: np.linalg.norm(
        centroides[i] - centro_roi
    )
)

# Mascara aproximada de la pupila
mascara_inicial = (
    etiquetas == componente_pupila
).astype(np.uint8)

# Limpiar pequenas irregularidades
kernel_ws = np.ones((3, 3), np.uint8)

mascara_inicial = cv2.morphologyEx(
    mascara_inicial,
    cv2.MORPH_CLOSE,
    kernel_ws
)

# Crear marcadores para Watershed
# 0 = zona desconocida
# 1 = fondo
# 2 = pupila

marcadores = np.zeros_like(
    imagen_ws, dtype=np.int32
)

# Definir fondo seguro mediante dilatacion
region_dilatada = cv2.dilate(
    mascara_inicial, kernel_ws, iterations=3
)

marcadores[region_dilatada == 0] = 1

# Definir pupila segura mediante erosion
pupila_segura = cv2.erode(
    mascara_inicial, kernel_ws, iterations=2
)

marcadores[pupila_segura == 1] = 2

# Convertir la imagen a BGR para cv2.watershed
imagen_ws_bgr = cv2.cvtColor(
    imagen_ws_filtrada, cv2.COLOR_GRAY2BGR
)

# Aplicar Watershed
marcadores_resultado = cv2.watershed(
    imagen_ws_bgr, marcadores.copy()
)

# Obtener mascara final de la pupila
mascara_watershed = (
    marcadores_resultado == 2
).astype(np.uint8) * 255

# Visualizar resultados
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

ax[0].imshow(imagen_ws, cmap="gray")
ax[0].set_title("ROI original")

ax[1].imshow(
    mascara_inicial, cmap="gray",
    vmin=0, vmax=1
)
ax[1].set_title("Mascara inicial")

ax[2].imshow(
    mascara_watershed, cmap="gray",
    vmin=0, vmax=255
)
ax[2].set_title("Mascara Watershed")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# 2.1.5 Comparacion de los metodos mediante Dice

# A. Crear o cargar la mascara de referencia manual

REFERENCIA_PATH = CARPETA_PROYECTO / "imagenes" / "mascara_referencia.png"

if REFERENCIA_PATH.exists():

    mascara_referencia = cv2.imread(
        str(REFERENCIA_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if mascara_referencia.shape != roi_gris.shape:
        raise ValueError(
            "La mascara de referencia no tiene las dimensiones de la ROI"
        )

    print("Se cargo la mascara de referencia guardada")

else:

    # Mostrar la ROI para marcar manualmente la pupila
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(roi_gris, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        "Marcar puntos alrededor del contorno pupilar\n"
        "Presionar Enter para finalizar"
    )

    print("Hace clic alrededor del borde real de la pupila.")
    print("Presiona Enter cuando termines.")

    puntos = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(puntos) < 3:
        raise ValueError(
            "Se necesitan al menos 3 puntos para definir el contorno"
        )

    # Convertir los puntos seleccionados a coordenadas de pixeles
    puntos = np.round(puntos).astype(np.int32)

    # Crear mascara binaria manual
    mascara_referencia = np.zeros_like(
        roi_gris, dtype=np.uint8
    )

    cv2.fillPoly(
        mascara_referencia,
        [puntos],
        255
    )

    # Guardar para reutilizar la misma referencia
    REFERENCIA_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    cv2.imwrite(
        str(REFERENCIA_PATH),
        mascara_referencia
    )


# B. Funcion para calcular Dice

def calcular_dice(mascara1, mascara2):

    # Convertir ambas mascaras a booleanas
    m1 = mascara1 > 0
    m2 = mascara2 > 0

    interseccion = np.logical_and(m1, m2).sum()

    total = m1.sum() + m2.sum()

    if total == 0:
        return 1.0

    dice = 2 * interseccion / total

    return dice


# C. Comparar los cuatro metodos

mascaras_segmentacion = {
    "Otsu": mascara_otsu,
    "K-Means K=2": mascaras_kmeans[2],
    "K-Means K=3": mascaras_kmeans[3],
    "K-Means K=4": mascaras_kmeans[4],
    "K-Means K=5": mascaras_kmeans[5],
    "Snakes": mascara_snake,
    "Watershed": mascara_watershed
}

resultados_dice = {}

print("\n--- Comparacion de segmentaciones mediante Dice ---")

for nombre, mascara in mascaras_segmentacion.items():

    valor_dice = calcular_dice(
        mascara_referencia,
        mascara
    )

    resultados_dice[nombre] = valor_dice

    print(f"{nombre}: Dice = {valor_dice:.4f}")


# D. Identificar el metodo con mayor Dice

mejor_metodo = max(
    resultados_dice,
    key=resultados_dice.get
)

mejor_mascara = mascaras_segmentacion[mejor_metodo].copy()

print(f"\nMejor metodo: {mejor_metodo}")
print(f"Dice obtenido: {resultados_dice[mejor_metodo]:.4f}")


# E. Visualizar la referencia y el mejor resultado

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(roi_gris, cmap="gray")
ax[0].set_title("ROI original")

ax[1].imshow(
    mascara_referencia,
    cmap="gray",
    vmin=0, vmax=255
)
ax[1].set_title("Referencia manual")

ax[2].imshow(
    mejor_mascara,
    cmap="gray",
    vmin=0, vmax=255
)
ax[2].set_title(f"Mejor metodo: {mejor_metodo}")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()



# 2.2.1 Operaciones morfologicas

# Utilizar la mascara con mayor Dice del apartado 2.1
mascara_base = mejor_mascara.copy()

print(f"Metodo seleccionado para morfologia: {mejor_metodo}")

# Elemento estructurante eliptico de 3x3
kernel_morf = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

# Erosion: reduce las regiones blancas
mascara_erosion = cv2.erode(
    mascara_base, kernel_morf, iterations=1
)

# Dilatacion: expande las regiones blancas
mascara_dilatacion = cv2.dilate(
    mascara_base, kernel_morf, iterations=1
)

# Apertura: erosion seguida de dilatacion
mascara_apertura = cv2.morphologyEx(
    mascara_base,
    cv2.MORPH_OPEN,
    kernel_morf
)

# Cierre: dilatacion seguida de erosion
mascara_cierre = cv2.morphologyEx(
    mascara_base,
    cv2.MORPH_CLOSE,
    kernel_morf
)

# Comparacion visual
imagenes_morf = [
    mascara_base,
    mascara_erosion,
    mascara_dilatacion,
    mascara_apertura,
    mascara_cierre
]

titulos_morf = [
    "Mascara original",
    "Erosion",
    "Dilatacion",
    "Apertura",
    "Cierre"
]

fig, ax = plt.subplots(1, 5, figsize=(16, 4))

for i in range(5):
    ax[i].imshow(
        imagenes_morf[i],
        cmap="gray",
        vmin=0,
        vmax=255
    )
    ax[i].set_title(titulos_morf[i])
    ax[i].axis("off")

plt.tight_layout()
plt.show()


# 2.2.2 Esqueletizacion de la pupila

from skimage.morphology import skeletonize

# Utilizar la mascara del mejor metodo de segmentacion
mascara_binaria = mascara_base > 0

# Aplicar esqueletizacion
esqueleto_pupila = skeletonize(mascara_binaria)

# Visualizar resultados
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(mascara_base, cmap="gray")
ax[0].set_title("Mascara pupilar original")

ax[1].imshow(esqueleto_pupila, cmap="gray")
ax[1].set_title("Esqueleto de la pupila")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()



# 2.2.3 Comparacion de mascaras mediante Dice e IoU

# Funcion para calcular IoU
def calcular_iou(mascara1, mascara2):

    m1 = mascara1 > 0
    m2 = mascara2 > 0

    interseccion = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0

    return interseccion / union


# Mascaras que vamos a comparar
mascaras_morfologicas = {
    "Sin morfologia": mascara_base,
    "Erosion": mascara_erosion,
    "Dilatacion": mascara_dilatacion,
    "Apertura": mascara_apertura,
    "Cierre": mascara_cierre
}

resultados_morfologia = {}

print("\n--- Comparacion de operaciones morfologicas ---")

for nombre, mascara in mascaras_morfologicas.items():

    # Reutilizamos la funcion Dice del apartado 2.1
    dice = calcular_dice(
        mascara_referencia,
        mascara
    )

    iou = calcular_iou(
        mascara_referencia,
        mascara
    )

    resultados_morfologia[nombre] = {
        "Dice": dice,
        "IoU": iou
    }

    print(f"\nOperacion: {nombre}")
    print(f"Dice = {dice:.4f}")
    print(f"IoU  = {iou:.4f}")


# Seleccionar la mascara con mayor Dice
mejor_operacion = max(
    resultados_morfologia,
    key=lambda nombre: resultados_morfologia[nombre]["Dice"]
)

mascara_final = mascaras_morfologicas[
    mejor_operacion
].copy()

print("\n--- Resultado final del Modulo 2 ---")
print(f"Metodo de segmentacion: {mejor_metodo}")
print(f"Operacion morfologica: {mejor_operacion}")
print(
    f"Dice final: "
    f"{resultados_morfologia[mejor_operacion]['Dice']:.4f}"
)
print(
    f"IoU final: "
    f"{resultados_morfologia[mejor_operacion]['IoU']:.4f}"
)


# Visualizar la mascara de referencia y la final
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(roi_gris, cmap="gray")
ax[0].set_title("ROI original")

ax[1].imshow(
    mascara_referencia,
    cmap="gray",
    vmin=0,
    vmax=255
)
ax[1].set_title("Referencia manual")

ax[2].imshow(
    mascara_final,
    cmap="gray",
    vmin=0,
    vmax=255
)
ax[2].set_title(
    f"{mejor_metodo} + {mejor_operacion}"
)

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


# Guardar la mascara final del Modulo 2

CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"

CARPETA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

RUTA_MASCARA_FINAL = (
    CARPETA_RESULTADOS / "mascara_pupila_final.png"
)

cv2.imwrite(
    str(RUTA_MASCARA_FINAL),
    mascara_final
)
print(f"\nMascara final guardada en: {RUTA_MASCARA_FINAL}")