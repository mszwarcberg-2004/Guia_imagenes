#Imports
from pathlib import Path
import cv2 #cv2: librería que permite abrir videos, extraer frames y procesar imagenes.
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from funciones_modulo1 import *
#Módulo 1:  

# Ruta del video
CARPETA_PROYECTO =CARPETA_PROYECTO = Path.cwd()
VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "video1.mov"

# Abrir el video
with VideoReader(str(VIDEO_PATH)) as video:
    total_frames = video.total_frames
    segundo = 6 #eligo un segundo particular de interés
    frame = video.get_frame_at_second(segundo) # Seleccionar un frame en un instante determinado

# Convertir de BGR (en lo que trabaja opencvs) a RGB para visualizarlo
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mostrar el frame
mostrar_figura((8,6),"Frame representativo del video",frame_rgb)

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
mostrar_figura((6,6),"Region de interes (ROI)",roi_rgb)

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

# Cuantizacion de la ROI a 4 y 2 bits por canal
roi_4bits = cuantizar(roi_rgb, 4)
roi_2bits = cuantizar(roi_rgb, 2)

# Visualizacion comparativa
mostrar_imagenes([roi_rgb,roi_4bits,roi_2bits],["Original - 8 bits","Cuantizacion - 4 bits","Cuantizacion - 2 bits"], figsize=(12, 4))

# 1.1.5 Espacios de color

# A. Separacion de los canales RGB
# Separar la ROI en sus tres canales de color

R, G, B = cv2.split(roi_rgb)

# Visualizar cada canal en escala de grises
mostrar_imagenes([R,G,B],["Canal R (Rojo)","Canal G (Verde)","Canal B (Azul)"], figsize=(12, 4),vmin=0,vmax=255)

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
imagenes = [frame_gris, frame_brillo, frame_contraste]
mostrar_imagenes(imagenes,["Frame original","Brillo+50","Contraste x1.5"],vmin=0,vmax=255)

# Calcular y comparar los histogramas

titulos = ["Histograma original", "Brillo +50", "Contraste x1.5"]
mostrar_histogramas(imagenes,titulos,log=True, figsize=(12, 4))

# 1.2.2 Escalado por rango de interes
# Convertir la ROI a escala de grises
roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#aplicamos funcion escalar por rango
roi_escalada,f1,f2 = escalar_por_rango(roi_gris)
print(f"f1 = {f1:.1f}")
print(f"f2 = {f2:.1f}")

# Visualizar las imagenes
mostrar_imagenes([roi_gris,roi_escalada], ["ROI original",f"Escalado lineal [{f1:.1f}, {f2:.1f}]"],figsize=(10, 4),vmin=0,vmax=255)

# Comparar los histogramas
mostrar_histogramas([roi_gris,roi_escalada],["Histograma original","Histograma escalado"],figsize=(10, 4))

# 1.2.3 Ecualización del histograma

# Ecualizacion global
roi_ecualizada = cv2.equalizeHist(roi_gris)

# Ecualizacion local (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
roi_clahe = clahe.apply(roi_gris)

# Comparar las imagenes

imagenes = [roi_gris, roi_escalada, roi_ecualizada, roi_clahe]
titulos = ["ROI original", "Escalado lineal", "Ecualizacion global", "CLAHE"]
mostrar_imagenes(imagenes,titulos,figsize=(16, 4),vmin=0, vmax=255)


# Comparar los histogramas
mostrar_histogramas(imagenes,titulos,figsize=(16,4))

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

mostrar_imagenes(imagenes,titulos,vmin=0,vmax=255,figsize=(16, 4))

# B. Simulacion de los distintos tipos de ruido

# Creamos un generador de numeros aleatorios
rng = np.random.default_rng(42)
#Agregamos el ruido usando funciones en el módulo 
roi_gaussiana = agregar_ruido_gaussiano(roi_gris,rng)
roi_sal_pimienta = agregar_sal_pimienta(roi_gris,rng)
roi_speckle = agregar_speckle(roi_gris, rng)

# Comparacion de los ruidos
imagenes_ruido = [roi_gris, roi_gaussiana, roi_sal_pimienta, roi_speckle]
titulos_ruido = ["ROI original", "Ruido gaussiano", "Sal y pimienta", "Ruido speckle"]

mostrar_imagenes(imagenes_ruido,titulos_ruido,vmin=0,vmax=255, figsize=(16, 4))

# C. Aplicacion de filtros espaciales

# Aplicar filtro gaussiano al ruido gaussiano
roi_gaussiana_filtrada = cv2.GaussianBlur(roi_gaussiana, (3, 3), 0)

# Aplicar filtro de mediana al ruido sal y pimienta
roi_sal_pimienta_filtrada = cv2.medianBlur(roi_sal_pimienta, 3)

# Aplicar filtro de media al ruido speckle
roi_speckle_filtrada = cv2.blur(roi_speckle, (3, 3))

# Comparar imagenes antes y despues del filtrado
imagenes = [roi_gaussiana, roi_gaussiana_filtrada, roi_sal_pimienta, roi_sal_pimienta_filtrada, roi_speckle, roi_speckle_filtrada]
titulos = ["Ruido gaussiano simulado", "Filtro gaussiano", "Ruido sal y pimienta simulado", "Filtro de mediana", "Ruido speckle simulado", "Filtro de media"]

mostrar_imagenes(imagenes,titulos,filas=3,figsize=(8,6), vmin=0,vmax=255)

# 1.3 Deteccion de bordes y dominio frecuencial

# 1.3.1 Deteccion de bordes

# Imagen original y filtrada en el punto 1.2.4
imagen_original = roi_gris
imagen_filtrada = roi_filtro_gaussiano

# Aplicar los operadores a ambas imagenes
bordes_original = detectar_bordes(imagen_original)
bordes_filtrada = detectar_bordes(imagen_filtrada)

nombres = ["Roberts", "Sobel", "Prewitt",
           "Laplaciano", "Canny"]

# Usamos la misma escala por operador para comparar original y filtrada
limites = [max(np.max(orig), np.max(filt), 1) for orig, filt in zip(bordes_original, bordes_filtrada)]

# Fila 1: ROI original + sus 5 bordes
# Fila 2: ROI filtrada + sus 5 bordes
imagenes = [imagen_original] + bordes_original + [imagen_filtrada] + bordes_filtrada

titulos = (["ROI original"] + [f"{n} - Original" for n in nombres]+ ["ROI filtrada - Gaussiano"] + [f"{n} - Filtrada" for n in nombres])

vmins = [None] + [0] * 5 + [None] + [0] * 5
vmaxs = [None] + limites + [None] + limites

mostrar_imagenes(imagenes, titulos, filas=2, figsize=(18, 7), vmin=vmins, vmax=vmaxs)

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
mostrar_imagenes([roi_gris, magnitud_log, fase_original],["ROI original", "Magnitud (log)", "Fase"], figsize=(12, 4), vmin=[None, None, -np.pi], vmax=[None, None, np.pi])

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

imagenes = [roi_gris, recon_magnitud, recon_fase, recon_cruzada_1, recon_cruzada_2]

imagenes=[normalizar_imagen(img) for img in imagenes]
titulos = ["ROI original", "Solo magnitud", "Solo fase", "Magnitud original + fase gaussiana", "Magnitud gaussiana + fase original"]

mostrar_imagenes(imagenes, titulos, figsize=(18, 4))

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
escala_maxima = max(np.max(psd_original_log), np.max(psd_gaussiana_log))

# Visualizar las PSD
mostrar_imagenes([psd_original_log,psd_gaussiana_log],["PSD - ROI original","PSD - ROI filtrada (Gaussiano)"],figsize=(10, 4),vmin=0,vmax=escala_maxima)

# 1.3.4 Filtrado en el dominio frecuencial
#Usamos la función de filtrado frecuencial definida en el módulo
frame_pasabajos, frame_pasaaltos, H_pasabajos, H_pasaaltos = filtrar_frecuencial(frame_gris)

# Aplicar filtro gaussiano en el dominio espacial
# para comparar ambos tipos de filtrado
frame_gaussiano = cv2.GaussianBlur(
    frame_gris, (5, 5), 0
)
# Visualizar las imagenes
titulos = ["Frame original", "Pasa-bajos frecuencial", "Pasa-altos frecuencial", "Gaussiano espacial"]
imagenes = [frame_gris, frame_pasabajos, frame_pasaaltos, frame_gaussiano]

# El pasa-altos puede contener valores negativos.
# Centramos la visualizacion en cero para observar bordes.
limite = max(np.max(np.abs(frame_pasaaltos)), 1)

mostrar_imagenes(imagenes, titulos, figsize=(16, 4), vmin=[0, 0, -limite, 0], vmax=[255, 255, limite, 255])

# Visualizar las mascaras frecuenciales
mostrar_imagenes([H_pasabajos,H_pasaaltos],["Mascara pasa-bajos","Mascara pasa-altos"],figsize=(8, 4))

# Preguntas finales del 1.3 - PSNR y SSIM

# Imagen de referencia: ROI original
imagen_referencia = roi_gris

# Imagenes obtenidas con los filtros espaciales del 1.2.4
imagenes_filtradas = {"Media": roi_media, "Mediana": roi_mediana, "Gaussiano": roi_filtro_gaussiano}

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
umbral, mascara_otsu = cv2.threshold(imagen_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
print(f"Umbral calculado por Otsu: {umbral}")

# Visualizar la ROI y la mascara obtenida
mostrar_imagenes([imagen_otsu, mascara_otsu], ["ROI original", "Segmentacion mediante Otsu"], figsize=(10, 4), vmin=[None, 0], vmax=[None, 255])

# 2.1.2 Segmentacion mediante K-Means
# Trabajamos con la ROI original en escala de grises
imagen_kmeans = roi_gris

# Convertir la imagen en una lista de pixeles
pixeles = imagen_kmeans.reshape(-1, 1).astype(np.float32)

# Criterio de finalizacion del algoritmo
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

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