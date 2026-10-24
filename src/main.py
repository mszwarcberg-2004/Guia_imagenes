#Imports
from pathlib import Path
import cv2 #cv2: librería que permite abrir videos, extraer frames y procesar imagenes.
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from funciones_modulo1 import *
from funciones_modulo2 import *
from visualizacion import *
from funciones_modulo3 import modulo3
from funciones_modulo4 import analizar_imagen, comparar_diametro_jpeg

#Constantes
CARPETA_PROYECTO = Path.cwd()
VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "video5.mp4"
CARPETA_IMAGENES = CARPETA_PROYECTO / "imagenes"
ROI_PATH = CARPETA_IMAGENES / "roi.png"
REFERENCIA_PATH = CARPETA_IMAGENES / "mascara_referencia.png"
SEGUNDO = 6 #segundo de interés del video
#Módulo 1:  
def modulo1():
    # Abrir el video
    with VideoReader(str(VIDEO_PATH)) as video:
        total_frames = video.total_frames
        segundo = SEGUNDO
        frame = video.get_frame_at_second(segundo) # Seleccionar un frame en un instante determinado

    # Convertir de BGR (en lo que trabaja opencvs) a RGB para visualizarlo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mostrar el frame
    mostrar_figura((8,6),"Frame representativo del video",frame_rgb)

    # Crear carpeta IMAGENES, donde vamos a guardar el frame y el roi
    CARPETA_IMAGENES.mkdir(exist_ok=True)

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

    # Guardar frame completo como png en mi compu
    cv2.imwrite(str(CARPETA_IMAGENES / "frame.png"), frame)

    # Visualizar ROI
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mostrar_figura((6,6),"Region de interes (ROI)",roi_rgb)

    # 1.1.2 Dimensiones y profundidad de bits
    alto, ancho, canales = roi.shape

    # Cantidad de bits por canal
    bits = roi.dtype.itemsize * 8
    print(f"Dimensiones de la ROI:\nAncho: {ancho} pixeles\nAlto: {alto} pixeles\nCantidad de canales: {canales}\nProfundidad de bits por canal: {bits}\nProfundidad total por pixel: {bits * canales} bits")

    # 1.1.3 Resolucion espacial: submuestreo
    escalas = [0.50, 0.25, 0.05]

    reducidas = [submuestrear(roi_rgb, e) for e in escalas]
    imagenes = [roi_rgb] + reducidas
    titulos = ["ROI original (100%)"] + [f"Resolucion {int(e * 100)}% ({img.shape[1]}x{img.shape[0]})" for e, img in zip(escalas, reducidas)]

    mostrar_imagenes(imagenes, titulos, filas=2, figsize=(12, 8), interpolation="nearest")

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
    mostrar_imagenes([H, S, V], ["H - Matiz", "S - Saturacion", "V - Valor"], figsize=(12, 4), vmin=0, vmax=[179, 255, 255])

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

    roi_media = cv2.blur(roi_gris, (3, 3)) # Filtro de media
    roi_mediana = cv2.medianBlur(roi_gris, 3) # Filtro de mediana
    roi_filtro_gaussiano = cv2.GaussianBlur(roi_gris, (3, 3), 0) # Filtro gaussiano

    # Comparar la imagen original con las tres filtradas
    imagenes = [roi_gris, roi_media, roi_mediana, roi_filtro_gaussiano]
    titulos = ["ROI original", "Filtro de media", "Filtro de mediana", "Filtro gaussiano"]

    mostrar_imagenes(imagenes,titulos,vmin=0,vmax=255,figsize=(16, 4))

    # B. Simulacion de los distintos tipos de ruido

    # Creamos un generador de numeros aleatorios
    rng = np.random.default_rng(42)
    #Agregamos el ruido usando funciones en el modulo
    roi_gaussiana = agregar_ruido_gaussiano(roi_gris,rng) #Ruido gaussiano
    roi_sal_pimienta = agregar_sal_pimienta(roi_gris,rng) #Ruido sal pimienta
    roi_speckle = agregar_speckle(roi_gris, rng) #Ruido speckle

    # Comparacion de los ruidos
    imagenes_ruido = [roi_gris, roi_gaussiana, roi_sal_pimienta, roi_speckle]
    titulos_ruido = ["ROI original", "Ruido gaussiano", "Sal y pimienta", "Ruido speckle"]

    mostrar_imagenes(imagenes_ruido,titulos_ruido,vmin=0,vmax=255, figsize=(16, 4))

    # C. Aplicacion de filtros espaciales

    roi_gaussiana_filtrada = cv2.GaussianBlur(roi_gaussiana, (3, 3), 0) # Filtro gaussiano al ruido gaussiano
    roi_sal_pimienta_filtrada = cv2.medianBlur(roi_sal_pimienta, 3) # Filtro de mediana al ruido de sal pimienta
    roi_speckle_filtrada = cv2.blur(roi_speckle, (3, 3)) # Filtro de media al ruido speckle

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

    nombres = ["Roberts", "Sobel", "Prewitt", "Laplaciano", "Canny"]

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
    F_gaussiana = np.fft.fftshift(np.fft.fft2(roi_filtro_gaussiano))

    magnitud_gaussiana = np.abs(F_gaussiana)
    fase_gaussiana = np.angle(F_gaussiana)

    # Magnitud original + fase gaussiana
    F_cruzada_1 = magnitud_original * np.exp(1j * fase_gaussiana)
    recon_cruzada_1 = np.fft.ifft2(np.fft.ifftshift(F_cruzada_1)).real

    # Magnitud gaussiana + fase original
    F_cruzada_2 = magnitud_gaussiana * np.exp(1j * fase_original)
    recon_cruzada_2 = np.fft.ifft2(np.fft.ifftshift(F_cruzada_2)).real

    # D. Visualizacion comparativa
    imagenes = [roi_gris, recon_magnitud, recon_fase, recon_cruzada_1, recon_cruzada_2]

    imagenes=[normalizar_imagen(img) for img in imagenes]
    titulos = ["ROI original", "Solo magnitud", "Solo fase", "Magnitud original + fase gaussiana", "Magnitud gaussiana + fase original"]

    mostrar_imagenes(imagenes, titulos, figsize=(18, 4))

    # 1.3.3 Densidad espectral de potencia (PSD)

    # Calcular la transformada de Fourier de la ROI original y filtrada
    F_original_psd = np.fft.fftshift(np.fft.fft2(roi_gris.astype(float)))
    F_gaussiana_psd = np.fft.fftshift(np.fft.fft2(roi_filtro_gaussiano.astype(float)))

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
    frame_gaussiano = cv2.GaussianBlur(frame_gris, (5, 5), 0)

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
        ssim_valor = ssim(imagen_referencia, imagen, data_range=255 )
        print(f"\nFiltro: {nombre}\nPSNR: {psnr_valor:.2f} dB\nSSIM: {ssim_valor:.4f}")
    
    return frame, roi, roi_gris
# MODULO 2: Segmentacion y morfologia matematica de la pupila

def modulo2(roi_gris):
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

    # Guardar las mascaras para compararlas despues
    mascaras_kmeans = {}
    imagenes_agrupadas = []
    mascaras_oscuras = []

    valores_k = [2, 3, 4, 5]

    for k in valores_k:

        # Aplicar K-Means con la funcion del modulo
        imagen_agrupada, mascara, centros = segmentar_kmeans(imagen_kmeans, k)

        mascaras_kmeans[k] = mascara
        imagenes_agrupadas.append(imagen_agrupada)
        mascaras_oscuras.append(mascara)

        print(f"K={k}: centros = {np.sort(centros.flatten())}")

    # Visualizar imagen agrupada (fila 1) y mascara del grupo mas oscuro (fila 2)
    titulos_kmeans = (
        [f"K-Means K={k}" for k in valores_k]
        + [f"Grupo mas oscuro K={k}" for k in valores_k]
    )

    mostrar_imagenes(imagenes_agrupadas + mascaras_oscuras,titulos_kmeans,filas=2,figsize=(14, 7),vmin=0,vmax=255)

    # 2.1.3 Segmentacion mediante contornos activos (Snakes)

    # Aplicar contorno activo con la funcion del modulo
    mascara_snake, contorno_inicial, contorno_final = segmentar_snake(roi_gris)

    # Visualizar el contorno inicial y final
    mostrar_snake(roi_gris, mascara_snake, contorno_inicial, contorno_final)

    # 2.1.4 Segmentacion mediante Watershed

    # Aplicar Watershed con la funcion del modulo
    mascara_inicial, mascara_watershed = segmentar_watershed(roi_gris, mascara_otsu)

    # Visualizar resultados
    mostrar_imagenes([roi_gris, mascara_inicial, mascara_watershed], ["ROI original", "Mascara inicial", "Mascara Watershed"], figsize=(13, 4), vmin=[None, 0, 0], vmax=[None, 1, 255])

    # 2.1.5 Comparacion de los metodos mediante Dice

    # A. Crear o cargar la mascara de referencia manual
    REFERENCIA_PATH = CARPETA_PROYECTO / "imagenes" / "mascara_referencia.png"
    mascara_referencia = cargar_o_crear_referencia(REFERENCIA_PATH, roi_gris)

    # B. Comparar los cuatro metodos
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

    # C. Identificar el metodo con mayor Dice
    mejor_metodo = max(resultados_dice, key=resultados_dice.get)
    mejor_mascara = mascaras_segmentacion[mejor_metodo].copy()

    print(f"\nMejor metodo: {mejor_metodo}\nDice obtenido: {resultados_dice[mejor_metodo]:.4f}")

    # D. Visualizar la referencia y el mejor resultado

    mostrar_imagenes([roi_gris, mascara_referencia, mejor_mascara], ["ROI original", "Referencia manual", f"Mejor metodo: {mejor_metodo}"], figsize=(12, 4), vmin=[None, 0, 0], vmax=[None, 255, 255])

    # 2.2.1 Operaciones morfologicas

    # Utilizar la mascara con mayor Dice del apartado 2.1
    mascara_base = mejor_mascara.copy()

    print(f"Metodo seleccionado para morfologia: {mejor_metodo}")

    # Erosion, dilatacion, apertura y cierre con la funcion del modulo
    mascaras_morfologicas = aplicar_morfologia(mascara_base)

    # Comparacion visual
    mostrar_imagenes(list(mascaras_morfologicas.values()), ["Mascara original", "Erosion", "Dilatacion", "Apertura", "Cierre"], figsize=(16, 4), vmin=0, vmax=255)

    # 2.2.2 Esqueletizacion de la pupila

    # Utilizar la mascara del mejor metodo de segmentacion
    mascara_binaria = mascara_base > 0

    # Aplicar esqueletizacion
    esqueleto_pupila = skeletonize(mascara_binaria)

    # Visualizar resultados
    mostrar_imagenes([mascara_base, esqueleto_pupila], ["Mascara pupilar original", "Esqueleto de la pupila"], figsize=(10, 4))

    # 2.2.3 Comparacion de mascaras mediante Dice e IoU

    # Las funciones calcular_dice y calcular_iou estan en funciones_modulo2
    # Las mascaras a comparar son las de mascaras_morfologicas (2.2.1)

    resultados_morfologia = {}

    print("\n--- Comparacion de operaciones morfologicas ---")

    for nombre, mascara in mascaras_morfologicas.items():

        # Reutilizamos la funcion Dice del apartado 2.1
        dice = calcular_dice(mascara_referencia, mascara)
        iou = calcular_iou(mascara_referencia, mascara)

        resultados_morfologia[nombre] = {"Dice": dice, "IoU": iou}

        print(f"\nOperacion: {nombre}\nDice = {dice:.4f}\nIoU  = {iou:.4f}")

    # Seleccionar la mascara con mayor Dice
    mejor_operacion = max(resultados_morfologia, key=lambda nombre: resultados_morfologia[nombre]["Dice"])
    mascara_final = mascaras_morfologicas[mejor_operacion].copy()

    print("\n--- Resultado final del Modulo 2 ---")
    print(f"Metodo de segmentacion: {mejor_metodo}\nOperacion morfologica: {mejor_operacion}\nDice final: {resultados_morfologia[mejor_operacion]['Dice']:.4f}\nIoU final: {resultados_morfologia[mejor_operacion]['IoU']:.4f}")

    # Visualizar la mascara de referencia y la final
    mostrar_imagenes([roi_gris, mascara_referencia, mascara_final], ["ROI original", "Referencia manual", f"{mejor_metodo} + {mejor_operacion}"], figsize=(12, 4), vmin=[None, 0, 0], vmax=[None, 255, 255])

    # Guardar la mascara final del Modulo 2
    CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    RUTA_MASCARA_FINAL = (CARPETA_RESULTADOS / "mascara_pupila_final.png")

    cv2.imwrite(str(RUTA_MASCARA_FINAL), mascara_final)
    print(f"\nMascara final guardada en: {RUTA_MASCARA_FINAL}")

    return mascara_final, mejor_metodo, mejor_operacion
# Definimos rutina main
def main():

    # Modulo 1
    frame, roi, roi_gris = modulo1()
    
    # Modulo 2
    mascara_final, mejor_metodo, mejor_operacion = modulo2(roi_gris)

    # Modulo 3
    modulo3(
        frame,
        roi,
        roi_gris,
        mascara_final,
        mejor_metodo,
        mejor_operacion,
        VIDEO_PATH,
        CARPETA_PROYECTO
    )

      # MODULO 4: COMPRESION DE IMAGENES

    print("\n========== MODULO 4 ==========")

    # 4.1 y 4.2: Huffman, RLE y JPEG sobre la ROI
    analizar_imagen(roi_gris, nombre="ROI del ojo")

    # 4.1 y 4.2: Huffman, RLE y JPEG sobre la mascara pupilar
    analizar_imagen(mascara_final, nombre="Mascara pupilar")

    # 4.1 y 4.2: Huffman, RLE y JPEG sobre el frame completo
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    analizar_imagen(frame_gris, nombre="Frame completo")

    # 4.3: Efecto de JPEG sobre el diametro pupilar
    comparar_diametro_jpeg(roi_gris)

    print("\n========== FIN MODULO 4 ==========")


if __name__ == "__main__":
    
    main()
