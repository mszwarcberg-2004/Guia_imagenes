# ============================================================
# MODULO 3 - Extraccion de caracteristicas y dinamica temporal
# ============================================================
# Este archivo expone tres cosas:
#
#   modulo3(...)                  -> Modulo 3 completo para el modo "desarrollo"
#                                    (3.1 en el frame segmentado + 3.2 en video).
#   analizar_frame_segmentado(...)-> solo 3.1 (descriptores y textura).
#   procesar_registros(...)       -> solo 3.2: segmenta la pupila en el video,
#                                    mide la curva A(t)/D(t), detecta la luz,
#                                    calcula los parametros fisiologicos y arma
#                                    las tablas. No necesita ROI ni mascara
#                                    manual: es lo que usa el modo "evaluacion".
# ============================================================

from pathlib import Path
from collections import Counter
import re
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature import graycoprops
from skimage.filters import gaussian, sobel
from skimage.segmentation import active_contour, watershed

try:
    from IPython.display import display
except ImportError:          # fuera de notebook / sin IPython
    display = print

EXTENSIONES_VIDEO = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}

# Reutiliza las mediciones ya calculadas de los otros videos (carpeta detalle)
USAR_CACHE_OTROS = True

# True: procesa tambien los otros videos de la carpeta datos/Videos_ojo (para la
#       tabla de variabilidad entre registros). False: solo el video principal.
PROCESAR_TODOS_LOS_VIDEOS = False

# True: ademas guarda mediciones_<video>.csv, serie_procesada_<video>.csv y
#       parametros_fisiologicos_<video>.csv con los nombres y columnas de antes,
#       para que sigan funcionando los scripts que ya leian esos archivos.
GUARDAR_FORMATO_ANTERIOR = False

METODOS_VALIDOS = {
    "Otsu", "K-Means K=2", "K-Means K=3", "K-Means K=4", "K-Means K=5",
    "Snakes", "Watershed"
}
OPERACIONES_VALIDAS = {
    "Sin morfologia", "Erosion", "Dilatacion", "Apertura", "Cierre"
}


# ------------------------------------------------------------
# PARAMETROS DEL DETECTOR (relativos al tamano del frame / de la pupila)
# ------------------------------------------------------------

# PARAMETROS (todos relativos al tamano del frame o de la pupila,
# no a pixeles absolutos de una imagen en particular)
#Estos parámetros son valores que configuramos para que el algoritmo pueda encontrar y seguir la pupila automáticamente en el video.


# Ancho al que se reduce el frame solo para localizar el centro
ANCHO_DETECCION = 420

# Zona amplia (fracciones del frame: xmin, xmax, ymin, ymax) donde se busca
# la semilla. Es solo para arrancar: despues se usa la zona de la semilla.
ZONA_BUSQUEDA = (0.10, 0.90, 0.15, 0.85)

# Rango de radios pupilares posibles, como fraccion del ancho del frame
RADIO_MIN_REL = 0.012
RADIO_MAX_REL = 0.08

# Puntaje minimo del detector (contraste relativo, sin unidades)
PUNTAJE_MINIMO = 0.20

# Cuanto se penaliza que el disco candidato no sea uniforme
PESO_UNIFORMIDAD = 0.5

# Frames usados para estimar la semilla
N_MUESTRAS_SEMILLA = 15

# Tamano de la zona de seguimiento alrededor de la semilla (en radios)
FACTOR_ZONA_SEGUIMIENTO = 3.0

# Rango de radios permitido respecto del radio de la semilla
RELACION_RADIO_SEMILLA = (0.5, 1.9)

# La segmentacion local se limita a un disco de este tamano (en radios)
FACTOR_DISCO_LOCAL = 1.8

# Controles de calidad
CIRCULARIDAD_MIN = 0.55
RELACION_AREA_MAX = 1.60          # cambio maximo de area entre frames
DESPLAZAMIENTO_MAX_REL = 1.0      # desplazamiento maximo (en radios)

# Parametros de 3.2 (fisiologicos, limpieza de la serie y artefactos)

FRECUENCIA_VISUALIZACION = 10     # Hz de los frames de revision (3.2.4)

# Parametros fisiologicos
VENTANA_BASAL_S = 1.0             # ventana previa al estimulo para el area basal
VENTANA_CONSTRICCION_S = 4.0      # maximo tras el encendido (se corta antes si hay apagado)
VENTANA_REDILATACION_S = 4.0      # ventana tras el apagado
BASAL_RETROCESO_S = 2.0          # el area basal se busca hasta 2 s antes del estimulo
BASAL_MARGEN_S = 0.25            # sin contar los ultimos 0.25 s (destello / artefacto del estimulo)
VENTANAS_BASAL_S = (0.5, 1.0, 2.0)   # area de referencia previa: se prueba la ventana corta primero
ULTIMOS_REDILATACION_S = 0.5      # area final de redilatacion = mediana de los ultimos 0.5 s de la ventana
DURACION_MIN_REDILATACION_S = 0.5 # la ventana tras el apagado debe durar al menos esto
LATENCIA_MIN_FISIOLOGICA_S = 0.10 # latencias menores se marcan para revision
AMPLITUD_MIN_SOBRE_RUIDO = 4.0    # si la amplitud es menor que 4x el ruido, no se informa latencia
UMBRAL_ESCALON_LUZ = 0.20         # instante del estimulo = 1er cruce del 20% del escalon
FRACCION_INICIO_CONSTRICCION = 0.05
DURACION_SOSTENIDA_S = 0.10       # el descenso debe sostenerse este tiempo para marcar el inicio
VENTANA_EXTREMO_S = 0.25          # suavizado para buscar minimo/maximo
MIN_PROPORCION_VALIDA = 0.50      # proporcion minima de muestras validas en una ventana

# Limpieza de la serie y artefactos
VENTANA_FILTRO_LOCAL_S = 0.30     # filtro de mediana local
TOLERANCIA_FILTRO_LOCAL = 0.35    # desvio relativo maximo respecto de la mediana local
MARGEN_PARPADEO_S = 0.05          # margen descartado a cada lado de un parpadeo
MIN_TRAMO_VALIDO_S = 0.10         # tramos validos mas cortos que esto se descartan
MIN_DURACION_PARPADEO_S = 0.10    # tramos sin medicion >= esto se marcan como parpadeo
MAX_DURACION_PARPADEO_S = 1.00    # si dura mas, se marca "perdida_seguimiento"
SALTO_OCULAR_REL = 0.35           # salto de centro entre frames (en radios) = mov. ocular

# ------------------------------------------------------------
# ESTADO DEL MODULO (lo fija configurar() antes de procesar)
# ------------------------------------------------------------
METODO_VIDEO = None
OPERACION_VIDEO = None
ZONA_SEGUIMIENTO = None
RANGO_RADIOS_SEMILLA = None
MOSTRAR_FIGURAS = True
CARPETA_FIGURAS = Path("resultados") / "figuras"
CARPETA_DETALLE = Path("resultados") / "detalle"

# Registro de por que se descartan frames (para diagnosticar)
MOTIVOS_RECHAZO = Counter()


def configurar(metodo, operacion, carpeta_resultados, mostrar_figuras=True):
    """Fija el metodo de segmentacion, la morfologia y las carpetas de salida."""
    global METODO_VIDEO, OPERACION_VIDEO, MOSTRAR_FIGURAS
    global CARPETA_FIGURAS, CARPETA_DETALLE

    if metodo not in METODOS_VALIDOS:
        raise ValueError(
            f"Metodo de segmentacion desconocido: {metodo!r}. "
            f"Opciones: {sorted(METODOS_VALIDOS)}"
        )

    if operacion not in OPERACIONES_VALIDAS:
        raise ValueError(
            f"Operacion morfologica desconocida: {operacion!r}. "
            f"Opciones: {sorted(OPERACIONES_VALIDAS)}"
        )

    METODO_VIDEO = metodo
    OPERACION_VIDEO = operacion
    MOSTRAR_FIGURAS = bool(mostrar_figuras)

    carpeta_resultados = Path(carpeta_resultados)
    CARPETA_FIGURAS = carpeta_resultados / "figuras"
    CARPETA_DETALLE = carpeta_resultados / "detalle"



# ------------------------------------------------------------
# DETECCION Y SEGUIMIENTO DE LA PUPILA, PARAMETROS, GRAFICOS Y TABLAS
# ------------------------------------------------------------

def detectar_centro_pupila(gris, zona, rango_radios, n_radios=6):
    """
    Busca un disco oscuro, uniforme y rodeado de un anillo mas claro.

    zona: (xmin, xmax, ymin, ymax) en pixeles del frame original.
    rango_radios: (rmin, rmax) en pixeles del frame original.
    """

    alto_original, ancho_original = gris.shape

    # Reducir la resolucion solo para localizar el centro
    escala = min(1.0, ANCHO_DETECCION / ancho_original)

    ancho = max(1, int(round(ancho_original * escala)))
    alto = max(1, int(round(alto_original * escala)))

    imagen = cv2.resize(
        gris,
        (ancho, alto),
        interpolation=cv2.INTER_AREA
    )

    imagen = cv2.GaussianBlur(
        imagen,
        (3, 3),
        0
    ).astype(np.float32)

    imagen_cuadrado = imagen ** 2

    # Zona de busqueda en la imagen reducida
    xmin = int(max(0, zona[0] * escala))
    xmax = int(min(ancho, zona[1] * escala))

    ymin = int(max(0, zona[2] * escala))
    ymax = int(min(alto, zona[3] * escala))

    if xmax <= xmin or ymax <= ymin:
        return None

    Y, X = np.ogrid[:alto, :ancho]

    zona_busqueda = (
        (X >= xmin) &
        (X < xmax) &
        (Y >= ymin) &
        (Y < ymax)
    )

    # Radios a evaluar (en la imagen reducida)
    rmin = max(3, int(round(rango_radios[0] * escala)))
    rmax = max(rmin, int(round(rango_radios[1] * escala)))

    radios = np.unique(
        np.round(np.linspace(rmin, rmax, n_radios)).astype(int)
    )

    mejor = None
    mejor_puntaje = -np.inf

    for radio in radios:

        # Discos y anillos de referencia
        radio_externo = int(np.ceil(1.8 * radio))

        coordenadas = np.arange(
            -radio_externo,
            radio_externo + 1
        )

        yy, xx = np.meshgrid(
            coordenadas,
            coordenadas,
            indexing="ij"
        )

        dist = np.sqrt(xx**2 + yy**2)

        disco = (dist <= radio).astype(np.float32)

        anillo = (
            (dist >= 1.2 * radio) &
            (dist <= 1.8 * radio)
        ).astype(np.float32)

        disco /= disco.sum()
        anillo /= anillo.sum()

        # Media del disco, media del cuadrado (para el desvio) y media del anillo
        media_disco = cv2.filter2D(imagen, -1, disco)
        media_disco_2 = cv2.filter2D(imagen_cuadrado, -1, disco)
        media_anillo = cv2.filter2D(imagen, -1, anillo)

        desvio_disco = np.sqrt(
            np.maximum(media_disco_2 - media_disco**2, 0)
        )

        # Contraste RELATIVO: no depende de la exposicion del frame.
        # (+10 evita dividir por casi cero en frames muy oscuros)
        referencia = media_anillo + 10.0

        contraste_relativo = (media_anillo - media_disco) / referencia

        # Se penaliza un disco poco uniforme (pestanas, sombras, bordes)
        puntaje_mapa = (
            contraste_relativo
            - PESO_UNIFORMIDAD * desvio_disco / referencia
        )

        # No evaluar circulos que se salgan del frame
        margen_valido = radio_externo + 1

        zona_valida = zona_busqueda.copy()

        zona_valida[:margen_valido, :] = False
        zona_valida[-margen_valido:, :] = False
        zona_valida[:, :margen_valido] = False
        zona_valida[:, -margen_valido:] = False

        if not np.any(zona_valida):
            continue

        puntajes = np.where(
            zona_valida,
            puntaje_mapa,
            -np.inf
        )

        indice = np.unravel_index(
            np.argmax(puntajes),
            puntajes.shape
        )

        puntaje = float(puntajes[indice])

        if puntaje > mejor_puntaje:

            mejor_puntaje = puntaje

            mejor = {
                "cx": float(indice[1] / escala),
                "cy": float(indice[0] / escala),
                "radio": float(radio / escala),
                "puntaje": puntaje
            }

    if mejor is None:
        return None

    if mejor_puntaje < PUNTAJE_MINIMO:
        return None

    return mejor


def estimar_semilla(ruta_video, n_muestras=N_MUESTRAS_SEMILLA):

    cap = cv2.VideoCapture(str(ruta_video))

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir {ruta_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(
        0,
        max(total - 1, 0),
        n_muestras,
        dtype=int
    )

    candidatos = []
    alto_v, ancho_v = None, None

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ok, fr = cap.read()

        if not ok:
            continue

        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        alto_v, ancho_v = g.shape

        zona = (
            ZONA_BUSQUEDA[0] * ancho_v,
            ZONA_BUSQUEDA[1] * ancho_v,
            ZONA_BUSQUEDA[2] * alto_v,
            ZONA_BUSQUEDA[3] * alto_v
        )

        rango = (
            RADIO_MIN_REL * ancho_v,
            RADIO_MAX_REL * ancho_v
        )

        c = detectar_centro_pupila(g, zona, rango, n_radios=8)

        if c is not None:
            candidatos.append(c)

    cap.release()

    if len(candidatos) < 3:
        raise ValueError(
            "No se pudo estimar la semilla de la pupila: "
            "muy pocas detecciones. Revisar ZONA_BUSQUEDA, "
            "PUNTAJE_MINIMO o la calidad del video."
        )

    cxs = np.array([c["cx"] for c in candidatos])
    cys = np.array([c["cy"] for c in candidatos])
    rs = np.array([c["radio"] for c in candidatos])

    # Consenso: descartar candidatos lejos de la mediana
    mediana_x = np.median(cxs)
    mediana_y = np.median(cys)
    mediana_r = np.median(rs)

    cerca = np.hypot(cxs - mediana_x, cys - mediana_y) <= mediana_r

    if cerca.sum() < 3:
        raise ValueError(
            "Las detecciones de la semilla no son consistentes entre frames. "
            "Revisar ZONA_BUSQUEDA o PUNTAJE_MINIMO."
        )

    return {
        "cx": float(np.median(cxs[cerca])),
        "cy": float(np.median(cys[cerca])),
        "radio": float(np.median(rs[cerca])),
        "ancho": ancho_v,
        "alto": alto_v,
        "n_validas": int(cerca.sum()),
        "n_muestras": len(indices)
    }


def quitar_reflejos(roi_local, zona_disco, radio):
    """Rellena (inpainting) los reflejos puntuales del flash.

    Un reflejo dentro de la pupila crea un borde muy fuerte que atrae a
    Snakes/Watershed y deja agujeros en Otsu/K-Means. Se detectan como
    manchas pequenas entre los pixeles mas brillantes del entorno de la
    pupila y se rellenan con el valor de sus vecinos.
    """

    valores = roi_local[zona_disco]

    if valores.size < 50:
        return roi_local

    umbral = np.percentile(valores, 98)

    brillantes = (
        (roi_local >= umbral) & zona_disco
    ).astype(np.uint8)

    cantidad, etiquetas_b, estadisticas_b, _ = (
        cv2.connectedComponentsWithStats(brillantes, 8)
    )

    # Solo manchas chicas: una region grande no es un reflejo
    # (es esclerotica, parpado, etc.)
    area_maxima = 0.08 * np.pi * radio**2

    reflejos = np.zeros(roi_local.shape, dtype=np.uint8)

    for i in range(1, cantidad):
        if estadisticas_b[i, cv2.CC_STAT_AREA] <= area_maxima:
            reflejos[etiquetas_b == i] = 255

    if not np.any(reflejos):
        return roi_local

    reflejos = cv2.dilate(
        reflejos,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    return cv2.inpaint(roi_local, reflejos, 3, cv2.INPAINT_TELEA)


def elegir_componente_central(binaria, cx, cy):
    """De una mascara binaria, se queda con la componente conectada
    que contiene al centro candidato (o la mas cercana a el)."""

    binaria = (binaria > 0).astype(np.uint8)

    cantidad, etiquetas_cc, _, centroides_cc = (
        cv2.connectedComponentsWithStats(binaria, 8)
    )

    if cantidad <= 1:
        return None

    xi = int(round(cx))
    yi = int(round(cy))

    if (
        0 <= yi < etiquetas_cc.shape[0] and
        0 <= xi < etiquetas_cc.shape[1] and
        etiquetas_cc[yi, xi] > 0
    ):
        elegido = etiquetas_cc[yi, xi]

    else:
        elegido = min(
            range(1, cantidad),
            key=lambda i: np.hypot(
                centroides_cc[i][0] - cx,
                centroides_cc[i][1] - cy
            )
        )

    return (etiquetas_cc == elegido).astype(np.uint8) * 255


def segmentar_otsu_local(roi_local, cx, cy, zona_disco):

    # Otsu calculado solo con los pixeles del entorno de la pupila
    # (pupila + iris), para que el umbral separe pupila de iris
    # y no iris de esclerotica.
    valores = roi_local[zona_disco].reshape(-1, 1)

    if valores.shape[0] < 20:
        return None

    umbral_local, _ = cv2.threshold(
        valores,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binaria = (
        (roi_local <= umbral_local) & zona_disco
    ).astype(np.uint8) * 255

    return elegir_componente_central(binaria, cx, cy)


def segmentar_kmeans_local(roi_local, cx, cy, k, zona_disco):

    # K-Means solo con los pixeles del entorno de la pupila
    pix = roi_local[zona_disco].reshape(-1, 1).astype(np.float32)

    if pix.shape[0] < 10 * k:
        return None

    criterio_local = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, etiq, centros_k = cv2.kmeans(
        pix,
        k,
        None,
        criterio_local,
        3,
        cv2.KMEANS_PP_CENTERS
    )

    grupo_oscuro_local = np.argmin(centros_k.flatten())

    binaria = np.zeros(roi_local.shape, dtype=np.uint8)

    binaria[zona_disco] = (
        (etiq.flatten() == grupo_oscuro_local).astype(np.uint8) * 255
    )

    return elegir_componente_central(binaria, cx, cy)


def segmentar_snake_local(roi_local, cx, cy, radio, zona_disco):

    # Circulo inicial alrededor de la pupila y contorno activo
    # sobre la imagen suavizada.

    # IMPORTANTE: la fuerza de borde de Snakes depende del contraste
    # absoluto de la imagen. En frames oscuros (antes del estimulo o
    # al apagar la luz) el contraste es bajo y las fuerzas internas
    # (alpha, beta) colapsan el contorno. Se normaliza el contraste
    # local a [0, 1] usando los pixeles del entorno de la pupila.
    valores_disco = roi_local[zona_disco].astype(float)

    p_bajo = np.percentile(valores_disco, 1)
    p_alto = np.percentile(valores_disco, 99)

    roi_normalizada = np.clip(
        (roi_local.astype(float) - p_bajo) / max(p_alto - p_bajo, 1.0),
        0,
        1
    )

    img = gaussian(
        roi_normalizada,
        sigma=1
    )

    radio_ini = 1.5 * radio

    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)

    filas = np.clip(
        cy + radio_ini * np.sin(ang),
        0,
        roi_local.shape[0] - 1
    )

    columnas = np.clip(
        cx + radio_ini * np.cos(ang),
        0,
        roi_local.shape[1] - 1
    )

    inicial = np.array([filas, columnas]).T

    final = active_contour(
        img,
        inicial,
        alpha=0.015,
        beta=10,
        gamma=0.001,
        w_line=0,
        w_edge=1,
        max_num_iter=500,   # menos iteraciones que en 2.1.3 (el video tiene muchos frames)
        boundary_condition="periodic"
    )

    puntos = np.round(final[:, ::-1]).astype(np.int32)

    mascara_s = np.zeros(roi_local.shape, dtype=np.uint8)

    cv2.fillPoly(mascara_s, [puntos], 255)

    return mascara_s


def segmentar_watershed_local(roi_local, cx, cy, radio, zona_disco):

    # Watershed sobre el gradiente de Sobel con marcadores
    # (interior seguro de la pupila / exterior)
    imagen_normalizada = (
        roi_local.astype(np.float64) / 255.0
    )

    gradiente = sobel(imagen_normalizada)

    Yl, Xl = np.ogrid[:roi_local.shape[0], :roi_local.shape[1]]

    distancia_l = np.sqrt(
        (Xl - cx)**2 +
        (Yl - cy)**2
    )

    # Marcadores:
    # 1 = interior seguro de la pupila
    # 2 = region exterior
    marcadores_l = np.zeros(roi_local.shape, dtype=np.int32)

    marcadores_l[distancia_l <= 0.30 * radio] = 1
    marcadores_l[distancia_l >= 1.65 * radio] = 2

    if not np.any(marcadores_l == 1):
        return None

    if not np.any(marcadores_l == 2):
        return None

    etiquetas_l = watershed(
        gradiente,
        markers=marcadores_l,
        watershed_line=True
    )

    return (etiquetas_l == 1).astype(np.uint8) * 255


def aplicar_morfologia_video(mascara, operacion):

    # Mismo elemento estructurante que en 2.2.1
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    if operacion == "Erosion":
        return cv2.erode(mascara, kernel_v, iterations=1)

    if operacion == "Dilatacion":
        return cv2.dilate(mascara, kernel_v, iterations=1)

    if operacion == "Apertura":
        return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_v)

    if operacion == "Cierre":
        return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_v)

    # "Sin morfologia"
    return mascara.copy()


def rechazar(motivo):
    MOTIVOS_RECHAZO[motivo] += 1
    return None


def segmentar_pupila(gris, candidato):

    alto_frame, ancho_frame = gris.shape

    cx = candidato["cx"]
    cy = candidato["cy"]
    radio = candidato["radio"]

    # Extraer una ROI local alrededor del candidato
    margen = max(12, int(np.ceil(2.4 * radio)))

    x1 = max(0, int(cx - margen))
    x2 = min(ancho_frame, int(cx + margen + 1))

    y1 = max(0, int(cy - margen))
    y2 = min(alto_frame, int(cy + margen + 1))

    roi = gris[y1:y2, x1:x2].copy()

    if roi.size == 0:
        return rechazar("roi_vacia")

    cx_local = cx - x1
    cy_local = cy - y1

    # Entorno de la pupila (proporcional al radio candidato)
    Yl, Xl = np.ogrid[:roi.shape[0], :roi.shape[1]]

    zona_disco = (
        (Xl - cx_local)**2 + (Yl - cy_local)**2
    ) <= (FACTOR_DISCO_LOCAL * radio)**2

    # Quitar reflejos del flash antes de segmentar
    roi = quitar_reflejos(roi, zona_disco, radio)

    # Se usa el metodo elegido en el Modulo 2
    if METODO_VIDEO == "Otsu":
        mascara = segmentar_otsu_local(
            roi, cx_local, cy_local, zona_disco
        )

    elif METODO_VIDEO.startswith("K-Means"):
        k_video = int(METODO_VIDEO.split("=")[1])
        mascara = segmentar_kmeans_local(
            roi, cx_local, cy_local, k_video, zona_disco
        )

    elif METODO_VIDEO == "Snakes":
        mascara = segmentar_snake_local(
            roi, cx_local, cy_local, radio, zona_disco
        )

    elif METODO_VIDEO == "Watershed":
        mascara = segmentar_watershed_local(
            roi, cx_local, cy_local, radio, zona_disco
        )

    else:
        raise ValueError(f"Metodo de segmentacion desconocido: {METODO_VIDEO}")

    if mascara is None:
        return rechazar("metodo_sin_mascara")

    # Operacion morfologica elegida en el Modulo 2
    mascara = aplicar_morfologia_video(mascara, OPERACION_VIDEO)

    # La mascara no puede salirse del entorno de la pupila
    mascara = cv2.bitwise_and(
        mascara,
        zona_disco.astype(np.uint8) * 255
    )

    # Conservar el contorno principal
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return rechazar("sin_contorno")

    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    # Envolvente convexa: la pupila es convexa, asi que se rellenan
    # muescas producidas por reflejos del flash o pestanas.
    contorno = cv2.convexHull(contorno)

    mascara_limpia = np.zeros_like(mascara)

    cv2.drawContours(
        mascara_limpia,
        [contorno],
        -1,
        255,
        cv2.FILLED
    )

    # Controles de calidad
    area = int(np.count_nonzero(mascara_limpia))

    area_circulo = np.pi * radio**2

    if not (0.35 * area_circulo <= area <= 2.50 * area_circulo):
        return rechazar("area_fuera_de_rango")

    # Rechazar segmentaciones cortadas por los bordes
    if (
        np.any(mascara_limpia[0, :]) or
        np.any(mascara_limpia[-1, :]) or
        np.any(mascara_limpia[:, 0]) or
        np.any(mascara_limpia[:, -1])
    ):
        return rechazar("toca_borde")

    perimetro = cv2.arcLength(contorno, True)

    if perimetro <= 0:
        return rechazar("perimetro_cero")

    circularidad = 4 * np.pi * area / perimetro**2

    if not (CIRCULARIDAD_MIN <= circularidad <= 1.20):
        return rechazar("circularidad")

    momentos = cv2.moments(
        mascara_limpia,
        binaryImage=True
    )

    if momentos["m00"] == 0:
        return rechazar("momento_cero")

    centro_final_x = (
        x1 + momentos["m10"] / momentos["m00"]
    )

    centro_final_y = (
        y1 + momentos["m01"] / momentos["m00"]
    )

    # El centro segmentado debe coincidir con el candidato
    if np.hypot(
        centro_final_x - cx,
        centro_final_y - cy
    ) > 0.60 * radio:
        return rechazar("centro_desplazado")

    diametro = 2 * np.sqrt(area / np.pi)

    # Mascara en coordenadas del frame completo
    mascara_frame = np.zeros_like(gris, dtype=np.uint8)

    mascara_frame[y1:y2, x1:x2] = mascara_limpia

    return {
        "mascara": mascara_frame,
        "area": float(area),
        "diametro": float(diametro),
        "cx": float(centro_final_x),
        "cy": float(centro_final_y),
        "radio": float(diametro / 2),
        "circularidad": float(circularidad),
        "puntaje": float(candidato["puntaje"])
    }


def procesar_frame_pupila(gris, anterior=None):
    """Devuelve (resultado, candidato). Cualquiera puede ser None."""

    if anterior is None:

        # Busqueda en toda la zona de seguimiento
        zona = ZONA_SEGUIMIENTO
        rango = RANGO_RADIOS_SEMILLA
        n_radios = 6

    else:

        # Seguimiento local alrededor de la ultima deteccion,
        # sin salirse de la zona de la semilla
        m = 2.0 * anterior["radio"]

        zona = (
            max(ZONA_SEGUIMIENTO[0], anterior["cx"] - m),
            min(ZONA_SEGUIMIENTO[1], anterior["cx"] + m),
            max(ZONA_SEGUIMIENTO[2], anterior["cy"] - m),
            min(ZONA_SEGUIMIENTO[3], anterior["cy"] + m)
        )

        rango = (
            max(RANGO_RADIOS_SEMILLA[0], 0.70 * anterior["radio"]),
            min(RANGO_RADIOS_SEMILLA[1], 1.40 * anterior["radio"])
        )

        if rango[0] >= rango[1]:
            rango = RANGO_RADIOS_SEMILLA

        n_radios = 4

    candidato = detectar_centro_pupila(
        gris,
        zona,
        rango,
        n_radios=n_radios
    )

    if candidato is None:
        MOTIVOS_RECHAZO["sin_candidato"] += 1
        return None, None

    resultado = segmentar_pupila(gris, candidato)

    if resultado is None:
        return None, candidato

    if anterior is not None:

        relacion_area = resultado["area"] / anterior["area"]

        # Rechazar variaciones excesivas entre frames
        if not (
            1.0 / RELACION_AREA_MAX <= relacion_area <= RELACION_AREA_MAX
        ):
            MOTIVOS_RECHAZO["salto_de_area_entre_frames"] += 1
            return None, candidato

        desplazamiento = np.hypot(
            resultado["cx"] - anterior["cx"],
            resultado["cy"] - anterior["cy"]
        )

        if desplazamiento > DESPLAZAMIENTO_MAX_REL * anterior["radio"]:
            MOTIVOS_RECHAZO["salto_de_posicion_entre_frames"] += 1
            return None, candidato

    return resultado, candidato


def cerrar_figura(fig, nombre, mostrar=None):
    """Guarda la figura en resultados/figuras y (opcional) la muestra."""
    if mostrar is None:
        mostrar = MOSTRAR_FIGURAS
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARPETA_FIGURAS / f"{nombre}.png", dpi=130, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def zonas_desde_semilla(sem):
    """Misma logica que se usaba para la semilla del video principal."""
    m = FACTOR_ZONA_SEGUIMIENTO * sem["radio"]

    zona = (
        max(0.0, sem["cx"] - m),
        min(float(sem["ancho"]), sem["cx"] + m),
        max(0.0, sem["cy"] - m),
        min(float(sem["alto"]), sem["cy"] + m)
    )

    rmin = max(RADIO_MIN_REL * sem["ancho"], RELACION_RADIO_SEMILLA[0] * sem["radio"])
    rmax = min(RADIO_MAX_REL * sem["ancho"], RELACION_RADIO_SEMILLA[1] * sem["radio"])

    if rmin >= rmax:
        rmin, rmax = 0.7 * sem["radio"], 1.4 * sem["radio"]

    return zona, (rmin, rmax)


def preparar_mascara_luz(shape, sem):
    """Anillo entre 2 y 3 radios de la semilla (fuera de la pupila)."""
    alto, ancho = shape
    r = sem["radio"]
    margen = int(np.ceil(3.2 * r))

    x1 = max(0, int(sem["cx"] - margen))
    x2 = min(ancho, int(sem["cx"] + margen + 1))
    y1 = max(0, int(sem["cy"] - margen))
    y2 = min(alto, int(sem["cy"] + margen + 1))

    Y, X = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((X - sem["cx"]) ** 2 + (Y - sem["cy"]) ** 2)
    mascara = (dist >= 2.0 * r) & (dist <= 3.0 * r)

    return (x1, x2, y1, y2, mascara)


def luminosidad_frame(gris, prep):
    x1, x2, y1, y2, mascara = prep
    roi = gris[y1:y2, x1:x2]
    # La mediana reduce la influencia de reflejos y pixeles saturados
    return float(np.median(roi[mascara])) if mascara.any() else np.nan


def tramos_bool(mascara):
    """Devuelve [(inicio, fin)] de los tramos consecutivos True."""
    mascara = np.asarray(mascara, dtype=bool)
    salida, i, n = [], 0, len(mascara)
    while i < n:
        if mascara[i]:
            j = i
            while j + 1 < n and mascara[j + 1]:
                j += 1
            salida.append((i, j))
            i = j + 1
        else:
            i += 1
    return salida


def _mediana_ventana(t, x, t0, t1):
    m = (t >= t0) & (t < t1) & np.isfinite(x)
    return float(np.median(x[m])) if m.sum() >= 3 else np.nan


def procesar_video(ruta, semilla=None, guardar_revision=False):
    """
    Segmenta la pupila frame a frame y mide la luminosidad en UNA sola
    lectura del video. Devuelve (DataFrame, fps_efectivo, revision, semilla).
    El eje de tiempo usa los timestamps reales del video (POS_MSEC) y,
    si no son confiables, cae a indice/fps.
    """
    global ZONA_SEGUIMIENTO, RANGO_RADIOS_SEMILLA

    ruta = Path(ruta)

    if semilla is None:
        semilla = estimar_semilla(ruta)

    zona_previa, rango_previo = ZONA_SEGUIMIENTO, RANGO_RADIOS_SEMILLA
    ZONA_SEGUIMIENTO, RANGO_RADIOS_SEMILLA = zonas_desde_semilla(semilla)

    MOTIVOS_RECHAZO.clear()
    cv2.setRNGSeed(0)   # K-Means reproducible

    cap = cv2.VideoCapture(str(ruta))

    try:
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir {ruta}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise ValueError("FPS invalido")

        paso = max(1, int(round(fps / FRECUENCIA_VISUALIZACION)))

        registros, revision, marcas_ms = [], {}, []
        anterior, fallos, idx = None, 0, 0
        prep_luz, caja = None, None

        while True:

            marca_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()

            if not ret:
                break

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prep_luz is None:
                prep_luz = preparar_mascara_luz(gris.shape, semilla)
                alto_f, ancho_f = gris.shape
                r0 = semilla["radio"]
                caja = (
                    max(0, int(ZONA_SEGUIMIENTO[0] - r0)),
                    min(ancho_f, int(ZONA_SEGUIMIENTO[1] + r0)),
                    max(0, int(ZONA_SEGUIMIENTO[2] - r0)),
                    min(alto_f, int(ZONA_SEGUIMIENTO[3] + r0))
                )

            resultado, candidato = procesar_frame_pupila(gris, anterior)

            if resultado is None:
                fallos += 1
                # Si se pierde el seguimiento, volver a buscar en la zona de la semilla
                if fallos >= 3:
                    anterior = None
                    resultado, candidato = procesar_frame_pupila(gris)

            if resultado is not None:
                anterior = resultado
                fallos = 0
                fila = {
                    "area_px2": resultado["area"],
                    "diametro_px": resultado["diametro"],
                    "centro_x": resultado["cx"],
                    "centro_y": resultado["cy"],
                    "circularidad": resultado["circularidad"],
                    "puntaje_deteccion": resultado["puntaje"],
                    "detectada": 1
                }
            else:
                fila = {
                    "area_px2": np.nan, "diametro_px": np.nan,
                    "centro_x": np.nan, "centro_y": np.nan,
                    "circularidad": np.nan, "puntaje_deteccion": np.nan,
                    "detectada": 0
                }

            fila["frame"] = idx
            fila["luminosidad"] = luminosidad_frame(gris, prep_luz)
            registros.append(fila)
            marcas_ms.append(marca_ms)

            # Frames de revision: se guarda solo el recorte alrededor del ojo
            # (no el frame completo) para no llenar la memoria.
            if guardar_revision and idx % paso == 0:
                bx1, bx2, by1, by2 = caja

                mascara = (
                    resultado["mascara"][by1:by2, bx1:bx2].copy()
                    if resultado is not None
                    else np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                )

                cand = None
                if candidato is not None:
                    cand = dict(candidato)
                    cand["cx"] -= bx1
                    cand["cy"] -= by1

                revision[idx] = (
                    frame[by1:by2, bx1:bx2].copy(),
                    mascara,
                    fila["area_px2"],
                    cand
                )

            idx += 1

    finally:
        cap.release()
        ZONA_SEGUIMIENTO, RANGO_RADIOS_SEMILLA = zona_previa, rango_previo

    if not registros:
        raise ValueError("No se procesaron frames")

    df = pd.DataFrame(registros)
    n = len(df)

    # Eje de tiempo: timestamps reales si son consistentes, si no indice/fps
    t_idx = np.arange(n) / fps
    t_ms = np.asarray(marcas_ms, dtype=float) / 1000.0

    usa_timestamps = (
        n > 2
        and np.all(np.isfinite(t_ms))
        and np.all(np.diff(t_ms) > 0)
        and abs((t_ms[-1] - t_ms[0]) - t_idx[-1]) < 0.2 * max(t_idx[-1], 1e-9)
    )

    tiempo = (t_ms - t_ms[0]) if usa_timestamps else t_idx
    df["tiempo_s"] = tiempo

    fps_efectivo = (n - 1) / tiempo[-1] if n > 1 and tiempo[-1] > 0 else fps

    print(
        f"  Eje de tiempo: "
        f"{'timestamps del video' if usa_timestamps else 'indice/fps'} | "
        f"fps nominal {fps:.2f}, efectivo {fps_efectivo:.2f} | "
        f"dif. maxima vs indice/fps: {np.max(np.abs(tiempo - t_idx)):.3f} s"
    )

    return df, float(fps_efectivo), revision, semilla


def filtro_hampel(valores, ventana=7, n_sigmas=3.0, piso_rel=0.05):
    """Marca como NaN los valores que se alejan demasiado de la mediana
    de su vecindad (picos aislados, segmentaciones erroneas)."""

    serie = pd.Series(valores, dtype=float)

    mediana = serie.rolling(ventana, center=True, min_periods=3).median()
    desvio_abs = (serie - mediana).abs()
    mad = desvio_abs.rolling(ventana, center=True, min_periods=3).median()

    umbral = np.maximum(n_sigmas * 1.4826 * mad, piso_rel * mediana)

    atipicos = (desvio_abs > umbral).to_numpy()

    salida = serie.copy()
    salida[atipicos] = np.nan

    return salida.to_numpy(), atipicos


def interpolar_huecos_cortos(valores, max_hueco=3):
    """Interpola solamente huecos de hasta max_hueco frames."""

    serie = pd.Series(valores, dtype=float)

    salida = serie.interpolate(method="linear", limit_area="inside")

    es_nan = serie.isna()
    grupos = es_nan.ne(es_nan.shift()).cumsum()

    for _, bloque in serie.groupby(grupos):
        if bloque.isna().all() and len(bloque) > max_hueco:
            salida.loc[bloque.index] = np.nan

    return salida.to_numpy().copy()


def detectar_transiciones_luz(t, lum, fps):
    """
    Devuelve (eventos, luminosidad_suave, umbral).
    eventos = [(tiempo_s, signo, escalon)] ordenados en el tiempo;
    signo +1 = luz sube (encendido), -1 = luz baja (apagado).
    El instante se toma como el primer cruce del 20% del escalon
    (no el maximo de la derivada, que cae a mitad de la subida).
    """

    lum = np.asarray(lum, dtype=float)

    if len(lum) < 10 or not np.any(np.isfinite(lum)):
        return [], np.full(len(lum), np.nan), np.nan

    vent = max(3, int(round(0.15 * fps)))
    if vent % 2 == 0:
        vent += 1

    ls = (
        pd.Series(lum)
        .interpolate(limit_direction="both")
        .rolling(vent, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    n = max(3, int(round(0.30 * fps)))

    cambios = np.full(len(ls), np.nan)

    for i in range(n, len(ls) - n):
        cambios[i] = np.median(ls[i:i + n]) - np.median(ls[i - n:i])

    validos = cambios[np.isfinite(cambios)]

    if validos.size == 0:
        return [], ls, np.nan

    mad = np.median(np.abs(validos - np.median(validos)))

    umbral = max(4 * 1.4826 * mad, 0.08 * np.median(ls), 5.0)

    # Candidatos: subidas y bajadas sostenidas
    candidatos = []

    for signo in (+1, -1):

        idx = np.where(signo * cambios > umbral)[0]

        if idx.size == 0:
            continue

        for g in np.split(idx, np.where(np.diff(idx) > n)[0] + 1):
            candidatos.append((int(g[np.argmax(signo * cambios[g])]), signo))

    # Refinar el instante de cada candidato
    eventos = []

    for pico, signo in sorted(candidatos):

        tp = t[pico]

        nivel_pre = _mediana_ventana(t, ls, tp - 0.8, tp - 0.3)
        nivel_post = _mediana_ventana(t, ls, tp + 0.3, tp + 0.8)

        if not (np.isfinite(nivel_pre) and np.isfinite(nivel_post)):
            continue

        escalon = nivel_post - nivel_pre

        if np.sign(escalon) != signo or abs(escalon) < umbral:
            continue

        nivel_cruce = nivel_pre + UMBRAL_ESCALON_LUZ * escalon

        zona = np.where((t >= tp - 0.6) & (t <= tp + 0.6))[0]
        cruces = zona[signo * (ls[zona] - nivel_cruce) >= 0]

        t_evento = float(t[cruces[0]]) if cruces.size else float(tp)

        eventos.append((t_evento, signo, float(escalon)))

    # Descartar escalones chicos (p. ej. el pico transitorio del flash al
    # decaer) y quedarse con transiciones que alternan de signo
    if eventos:
        maximo = max(abs(e[2]) for e in eventos)
        eventos = [e for e in eventos if abs(e[2]) >= 0.4 * maximo]

    depurados = []

    for e in sorted(eventos):
        if depurados and depurados[-1][1] == e[1]:
            if abs(e[2]) > abs(depurados[-1][2]):
                depurados[-1] = e
        else:
            depurados.append(e)

    return depurados, ls, float(umbral)


def marcar_eventos(t, area_limpia, cx, cy, radio_ref, fps):
    """
    Columna 'evento' por frame:
      - parpadeo: tramo sin medicion valida (no detectada o atipica) de
        MIN_DURACION_PARPADEO_S a MAX_DURACION_PARPADEO_S
      - perdida_seguimiento: mismo tramo pero mas largo
      - mov_ocular: salto del centro entre frames > SALTO_OCULAR_REL * radio
    """

    n = len(t)
    evento = np.full(n, "", dtype=object)

    min_frames = max(2, int(round(MIN_DURACION_PARPADEO_S * fps)))

    for i, j in tramos_bool(~np.isfinite(area_limpia)):

        if j - i + 1 < min_frames:
            continue

        duracion = (j - i + 1) / fps

        evento[i:j + 1] = (
            "parpadeo" if duracion <= MAX_DURACION_PARPADEO_S
            else "perdida_seguimiento"
        )

    salto = np.hypot(np.diff(cx), np.diff(cy))

    indices = np.where(
        np.isfinite(salto) & (salto > SALTO_OCULAR_REL * radio_ref)
    )[0] + 1

    for k in indices:
        if evento[k] == "":
            evento[k] = "mov_ocular"

    return evento


def mostrar_revision(revision, fps, nombre, mostrar=None):
    """3.2.4: frames de revision con el contorno detectado, en tandas."""

    indices = sorted(revision)
    POR_TANDA = 10

    for k, inicio in enumerate(range(0, len(indices), POR_TANDA)):

        tanda = indices[inicio:inicio + POR_TANDA]

        fig, ax = plt.subplots(2, len(tanda), figsize=(15, 3.5), squeeze=False)

        for j, indice in enumerate(tanda):

            frame_c, mascara, area, candidato = revision[indice]

            original_rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
            con_contorno = original_rgb.copy()

            contornos, _ = cv2.findContours(
                mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(con_contorno, contornos, -1, (0, 255, 0), 2)

            if candidato is not None:
                cv2.circle(
                    con_contorno,
                    (int(round(candidato["cx"])), int(round(candidato["cy"]))),
                    int(round(candidato["radio"])), (255, 0, 0), 1
                )

            ax[0, j].imshow(original_rgb)
            ax[0, j].set_title(f"{indice / fps:.2f} s", fontsize=8)

            ax[1, j].imshow(con_contorno)
            ax[1, j].set_title(
                f"{area:.0f} px²" if np.isfinite(area) else "No detectada",
                fontsize=8
            )

            ax[0, j].axis("off")
            ax[1, j].axis("off")

        plt.subplots_adjust(left=0.01, right=0.99, top=0.88,
                            bottom=0.05, wspace=0.08, hspace=0.25)

        cerrar_figura(fig, f"revision_{nombre}_{k + 1:02d}", mostrar)


def filtro_mediana_local(valores, fps):
    """Descarta valores que se alejan demasiado de la mediana de su
    entorno (0.3 s). Atrapa saltos de varios frames seguidos (por ejemplo,
    el borde del parpado al cerrarse) que el filtro de Hampel no detecta."""

    serie = pd.Series(valores, dtype=float)

    ventana = max(7, int(round(VENTANA_FILTRO_LOCAL_S * fps)))
    if ventana % 2 == 0:
        ventana += 1

    mediana = serie.rolling(ventana, center=True, min_periods=3).median()

    relativo = (serie - mediana).abs() / mediana

    salida = serie.copy()
    salida[(relativo > TOLERANCIA_FILTRO_LOCAL).to_numpy()] = np.nan

    return salida.to_numpy()


def limpiar_bordes_de_huecos(valores, fps):
    """Elimina mediciones poco confiables pegadas a un parpadeo:
    (1) un pequeno margen antes y despues de cada hueco largo y
    (2) tramos validos demasiado cortos (quedaron aislados)."""

    v = np.asarray(valores, dtype=float).copy()
    n = len(v)

    min_hueco = max(2, int(round(MIN_DURACION_PARPADEO_S * fps)))
    margen = max(2, int(round(MARGEN_PARPADEO_S * fps)))

    for i, j in tramos_bool(~np.isfinite(v)):
        if j - i + 1 >= min_hueco:
            v[max(0, i - margen):i] = np.nan
            v[j + 1:min(n, j + 1 + margen)] = np.nan

    min_tramo = max(5, int(round(MIN_TRAMO_VALIDO_S * fps)))

    for i, j in tramos_bool(np.isfinite(v)):
        if j - i + 1 < min_tramo:
            v[i:j + 1] = np.nan

    return v


def procesar_serie_area(areas, fps):
    """Hampel -> filtro de mediana local -> limpieza de bordes de parpadeo
    -> interpolacion de huecos de hasta 3 frames -> mediana de 5 frames.
    Devuelve (serie_limpia_con_NaN, descartados, serie_procesada)."""

    areas = np.asarray(areas, dtype=float)

    sin_atipicos, _ = filtro_hampel(areas)
    sin_atipicos = filtro_mediana_local(sin_atipicos, fps)
    limpia = limpiar_bordes_de_huecos(sin_atipicos, fps)

    interpolada = interpolar_huecos_cortos(limpia, max_hueco=3)

    procesada = (
        pd.Series(interpolada)
        .rolling(window=5, center=True, min_periods=3)
        .median()
        .to_numpy()
        .copy()
    )

    # Mantener los intervalos sin mediciones
    procesada[~np.isfinite(interpolada)] = np.nan

    descartados = np.isfinite(areas) & ~np.isfinite(limpia)

    return limpia, descartados, procesada


def _suavizar_extremo(valores, fps):
    """Mediana movil (0.25 s) para buscar minimos/maximos sin que un
    pico aislado de segmentacion se confunda con la respuesta real."""

    w = max(5, int(round(VENTANA_EXTREMO_S * fps)))
    if w % 2 == 0:
        w += 1

    suave = (
        pd.Series(valores)
        .rolling(w, center=True, min_periods=w // 2 + 1)
        .median()
        .to_numpy()
    )

    return suave if np.any(np.isfinite(suave)) else np.asarray(valores, float)


def _referencia_inmediata(t, a, t0):
    """Area de referencia = mediana de las mediciones validas justo antes de t0
    (se usa como referencia previa al apagado, para la redilatacion).
    Se empieza por una ventana corta (para no arrastrar tendencias previas) y
    se agranda si hay un parpadeo cerca. Devuelve (valor, inicio_de_ventana)."""

    for ventana in VENTANAS_BASAL_S:
        valor = _mediana_ventana(t, a, t0 - ventana, t0)
        if np.isfinite(valor):
            return valor, t0 - ventana

    return np.nan, t0 - VENTANAS_BASAL_S[-1]


def _basal_previo(t, a, t0):
    """Area basal previa al encendido (constriccion).

    Por defecto es la mediana de los 0.5 s inmediatamente anteriores al
    estimulo. Pero si esa ventana esta contaminada (un parpadeo, o el destello
    o el rebote de la segmentacion justo al encender), se usa en su lugar la
    ventana de 0.5 s mas "tranquila" (menor dispersion) de los 2 s previos, sin
    contar los ultimos 0.25 s. Entre las igual de tranquilas se elige la mas
    reciente.

    Devuelve (valor, inicio, fin) de la ventana usada.
    """

    ventana = VENTANAS_BASAL_S[0]

    # 1) Ventana inmediata
    m = (t >= t0 - ventana) & (t < t0) & np.isfinite(a)

    inmediata = None

    if m.sum() >= 5:
        x = a[m]
        mediana = float(np.median(x))
        dispersion = float(np.median(np.abs(x - mediana)))
        inmediata = (mediana, dispersion)

    # 2) Ventanas candidatas "tranquilas" dentro de los 2 s previos
    fin_max = t0 - BASAL_MARGEN_S
    inicio = max(t[0], t0 - BASAL_RETROCESO_S)

    candidatas = []

    for ini in np.arange(inicio, fin_max - ventana + 1e-9, 0.1):

        mk = (t >= ini) & (t < ini + ventana) & np.isfinite(a)

        if mk.sum() >= 5:
            x = a[mk]
            med = float(np.median(x))
            disp = float(np.median(np.abs(x - med)))
            candidatas.append((float(ini), med, disp))

    tranquila = None

    if candidatas:

        minima = min(c[2] for c in candidatas)

        elegibles = [
            c for c in candidatas
            if c[2] <= 1.5 * minima + 0.005 * abs(c[1])
        ]

        ini, med, _ = max(elegibles)

        tranquila = (med, ini, ini + ventana, minima)

    # 3) Decidir
    if inmediata is not None:

        usar_inmediata = (
            tranquila is None
            or inmediata[1] <= 2 * tranquila[3] + 0.02 * abs(inmediata[0])
        )

        if usar_inmediata:
            return inmediata[0], t0 - ventana, t0

    if tranquila is not None:
        return tranquila[0], tranquila[1], tranquila[2]

    # Respaldo: ventanas inmediatas mas largas
    valor, ini = _referencia_inmediata(t, a, t0)

    return valor, ini, t0


def _velocidad_pico(tr, ar, fps, signo):
    """Pendiente maxima (px2/s) medida sobre ~0.2 s, sin cruzar huecos de datos.
    signo = -1 para constriccion (area que baja), +1 para redilatacion."""

    k = max(2, int(round(0.2 * fps)))

    if len(tr) <= k:
        return np.nan

    dt = tr[k:] - tr[:-k]
    pendiente = signo * (ar[k:] - ar[:-k]) / dt
    sin_hueco = dt <= 1.5 * k / fps

    if not np.any(sin_hueco):
        return np.nan

    return float(np.max(pendiente[sin_hueco]))


def _constriccion(t, a, t_on, t_off, fps):

    hay_on = np.isfinite(t_on)

    if hay_on:
        t0 = t_on
        # Basal: mediana de lo medido justo antes del estimulo
        basal, t_base_ini, t_base_fin = _basal_previo(t, a, t0)
        fin = t0 + VENTANA_CONSTRICCION_S
        if np.isfinite(t_off):
            fin = min(fin, t_off)
    else:
        # Plan B: no se detecto la luz -> amplitud y velocidad pico, sin latencia
        t0 = t[0] + VENTANA_BASAL_S
        basal = _mediana_ventana(t, a, t[0], t0)
        fin = t[-1]
        t_base_ini = t[0]
        t_base_fin = t0

    if not np.isfinite(basal):
        return {"estado_constriccion": "Basal insuficiente"}

    en_ventana = (t >= t0) & (t <= fin)
    validos = en_ventana & np.isfinite(a)

    if validos.sum() < 5 or validos.sum() < MIN_PROPORCION_VALIDA * en_ventana.sum():
        return {
            "area_basal_px2": float(basal),
            "estado_constriccion": "Respuesta incompleta"
        }

    tr, ar = t[validos], a[validos]

    ar_suave = _suavizar_extremo(ar, fps)

    i_min = int(np.nanargmin(ar_suave))
    a_min = float(ar_suave[i_min])
    t_min = float(tr[i_min])
    amp = basal - a_min

    R = {
        "area_basal_px2": float(basal),
        "area_minima_px2": a_min,
        "tiempo_minimo_s": t_min
    }

    if amp <= 0:
        R["estado_constriccion"] = "Sin constriccion"
        return R

    R["amplitud_px2"] = float(amp)
    R["amplitud_pct"] = float(100 * amp / basal)

    # Ruido de la medicion antes del estimulo
    vb = a[(t >= t_base_ini) & (t < t_base_fin) & np.isfinite(a)]

    ruido = (
        1.4826 * np.median(np.abs(vb - np.median(vb)))
        if vb.size >= 3 else 0.0
    )

    # Velocidad pico de constriccion: pendiente maxima entre t0 y el minimo
    pico = _velocidad_pico(tr[:i_min + 1], ar[:i_min + 1], fps, -1)
    if np.isfinite(pico):
        R["velocidad_constriccion_pico_px2_s"] = pico

    # Si la amplitud es comparable al ruido, el inicio de la constriccion
    # no se puede ubicar con confianza: se informa amplitud, sin latencia.
    if ruido > 0 and amp < AMPLITUD_MIN_SOBRE_RUIDO * ruido:
        R["estado_constriccion"] = "Parcial (amplitud comparable al ruido: sin latencia)"
        return R

    # Umbral de inicio: 5% de la amplitud o 3 sigmas del ruido basal
    paso_inicio = min(
        max(FRACCION_INICIO_CONSTRICCION * amp, 3 * ruido),
        0.5 * amp
    )

    umbral_inicio = basal - paso_inicio

    n_sostenido = max(3, int(round(DURACION_SOSTENIDA_S * fps)))

    i_ini = None

    for i in range(i_min + 1):
        tramo = ar[i:i + n_sostenido]
        if len(tramo) == n_sostenido and np.all(tramo <= umbral_inicio):
            i_ini = i
            break

    if i_ini is not None:

        t_ini = float(tr[i_ini])
        R["tiempo_inicio_s"] = t_ini

        if hay_on:
            R["latencia_s"] = float(t_ini - t_on)

        if t_min - t_ini > 0:
            R["velocidad_constriccion_px2_s"] = float(amp / (t_min - t_ini))

        if not hay_on:
            R["estado_constriccion"] = "Parcial (luz no detectada: sin latencia)"
        elif R["latencia_s"] < LATENCIA_MIN_FISIOLOGICA_S:
            R["estado_constriccion"] = "Calculada (latencia < 0.1 s: revisar)"
        else:
            R["estado_constriccion"] = "Calculada"

    else:

        if hay_on and t_min > t_on:
            R["velocidad_constriccion_px2_s"] = float(amp / (t_min - t_on))

        R["estado_constriccion"] = (
            "Parcial (inicio no identificado: sin latencia)"
            if hay_on else
            "Parcial (luz no detectada: sin latencia)"
        )

    return R


def _redilatacion(t, a, t_off, ons, fps):

    # La redilatacion se mide desde el APAGADO, no desde el minimo.
    a_ref, _ = _referencia_inmediata(t, a, t_off)

    if not np.isfinite(a_ref):
        return {"estado_redilatacion": "Referencia insuficiente"}

    fin = min(t_off + VENTANA_REDILATACION_S, t[-1])

    siguientes = [x for x in ons if x > t_off]
    if siguientes:
        fin = min(fin, siguientes[0])

    en_ventana = (t >= t_off) & (t <= fin)
    validos = en_ventana & np.isfinite(a)

    R = {"area_ref_redilatacion_px2": float(a_ref)}

    if validos.sum() < 5 or validos.sum() < MIN_PROPORCION_VALIDA * en_ventana.sum():
        R["estado_redilatacion"] = "Respuesta incompleta"
        return R

    tr, ar = t[validos], a[validos]

    if tr[-1] - t_off < DURACION_MIN_REDILATACION_S:
        R["estado_redilatacion"] = "Respuesta insuficiente (video termina poco despues del apagado)"
        return R

    # Area al FINAL de la ventana: mediana de las ultimas mediciones validas.
    # No se usa el maximo de la ventana: un valor alto aislado cerca del apagado
    # (parpadeo, destello) daria un tiempo casi nulo y una velocidad absurda.
    cola = tr >= (tr[-1] - ULTIMOS_REDILATACION_S)
    a_fin = float(np.median(ar[cola]))
    t_fin = float(np.median(tr[cola]))
    amp = a_fin - a_ref

    R["area_max_redilatacion_px2"] = a_fin
    R["tiempo_max_redilatacion_s"] = t_fin

    if amp <= 0:
        R["estado_redilatacion"] = "Sin redilatacion"
        return R

    R["velocidad_redilatacion_px2_s"] = float(amp / (t_fin - t_off))

    pico = _velocidad_pico(tr, ar, fps, +1)
    if np.isfinite(pico):
        R["velocidad_redilatacion_pico_px2_s"] = pico

    R["estado_redilatacion"] = "Calculada"

    return R


def calcular_parametros(t, a, eventos, fps):
    """Clasifica el registro (A: la luz sube primero / B: la luz baja primero)
    y calcula parametros de constriccion (desde el encendido) y de
    redilatacion (desde el apagado). Siempre devuelve las mismas claves."""

    nan = np.nan

    P = {
        "tipo_registro": "Indeterminado",
        "tiempo_estimulo_s": nan, "tiempo_apagado_s": nan,
        "area_basal_px2": nan, "area_minima_px2": nan,
        "tiempo_inicio_s": nan, "tiempo_minimo_s": nan,
        "latencia_s": nan,
        "amplitud_px2": nan, "amplitud_pct": nan,
        "velocidad_constriccion_px2_s": nan,
        "velocidad_constriccion_pico_px2_s": nan,
        "area_ref_redilatacion_px2": nan,
        "area_max_redilatacion_px2": nan,
        "tiempo_max_redilatacion_s": nan,
        "velocidad_redilatacion_px2_s": nan,
        "velocidad_redilatacion_pico_px2_s": nan,
        "estado_constriccion": "No calculada",
        "estado_redilatacion": "No calculada"
    }

    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)

    ons = [e[0] for e in eventos if e[1] > 0]
    offs_todos = [e[0] for e in eventos if e[1] < 0]

    t_on = ons[0] if ons else nan

    offs = [x for x in offs_todos if (not np.isfinite(t_on)) or x > t_on]
    t_off = offs[0] if offs else nan

    P["tiempo_estimulo_s"] = t_on
    P["tiempo_apagado_s"] = t_off

    if eventos:
        P["tipo_registro"] = "A" if eventos[0][1] > 0 else "B"

    sin_luz = len(eventos) == 0

    if np.isfinite(t_on) or sin_luz:
        P.update(_constriccion(t, a, t_on, t_off, fps))
    else:
        P["estado_constriccion"] = "No aplica (sin encendido)"

    if np.isfinite(t_off):
        P.update(_redilatacion(t, a, t_off, ons, fps))
    elif sin_luz:
        P["estado_redilatacion"] = "No aplica (luz no detectada)"
    else:
        P["estado_redilatacion"] = "No aplica (sin apagado)"

    return P


def analizar_registro(df, fps):

    t = df["tiempo_s"].to_numpy(float)
    areas = df["area_px2"].to_numpy(float)

    limpia, descartados, a_proc = procesar_serie_area(areas, fps)
    d_proc = 2 * np.sqrt(a_proc / np.pi)

    radio_ref = np.nanmedian(df["diametro_px"].to_numpy(float)) / 2

    # Los parpadeos se marcan sobre la medicion cruda (frames donde NO se
    # detecto pupila), no sobre la serie ya filtrada.
    evento = marcar_eventos(
        t, areas,
        df["centro_x"].to_numpy(float),
        df["centro_y"].to_numpy(float),
        radio_ref, fps
    )

    # La luminosidad medida durante un parpadeo no sirve (el parpado tapa
    # el ojo): se descarta y se interpola antes de buscar el estimulo.
    lum = df["luminosidad"].to_numpy(float).copy()
    margen = max(2, int(round(MARGEN_PARPADEO_S * fps)))

    for etiqueta in ("parpadeo", "perdida_seguimiento"):
        for i, j in tramos_bool(evento == etiqueta):
            lum[max(0, i - margen):j + margen + 1] = np.nan

    eventos_luz, lum_suave, _ = detectar_transiciones_luz(t, lum, fps)

    params = calcular_parametros(t, a_proc, eventos_luz, fps)

    out = pd.DataFrame({
        "frame": df["frame"].to_numpy(),
        "tiempo_s": t,
        "area_original_px2": areas,
        "area_procesada_px2": a_proc,
        "diametro_original_px": df["diametro_px"].to_numpy(float),
        "diametro_procesado_px": d_proc,
        "centro_x": df["centro_x"].to_numpy(float),
        "centro_y": df["centro_y"].to_numpy(float),
        "circularidad": df["circularidad"].to_numpy(float),
        "detectada": df["detectada"].to_numpy(),
        "luminosidad": df["luminosidad"].to_numpy(float),
        "luminosidad_suave": lum_suave,
        "descartado": descartados.astype(int),
        "evento": evento
    })

    conteo = {
        "n_parpadeos": len(tramos_bool(evento == "parpadeo")),
        "n_perdidas_seguimiento": len(tramos_bool(evento == "perdida_seguimiento")),
        "n_mov_oculares": len(tramos_bool(evento == "mov_ocular")),
        "n_descartados": int(descartados.sum())
    }

    return out, params, eventos_luz, conteo


def graficar_curvas(out, nombre, mostrar):
    """3.2.2: area y diametro en funcion del tiempo."""

    t = out["tiempo_s"].to_numpy()
    area_o = out["area_original_px2"].to_numpy()
    area_p = out["area_procesada_px2"].to_numpy()
    diam_o = out["diametro_original_px"].to_numpy()
    diam_p = out["diametro_procesado_px"].to_numpy()
    descartados = out["descartado"].to_numpy() == 1

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax[0].plot(t, area_o, color="gray", alpha=0.5, label="Medicion original")

    if descartados.any():
        ax[0].scatter(
            t[descartados], area_o[descartados],
            color="red", s=14, zorder=3, label="Atipicos descartados"
        )

    ax[0].plot(t, area_p, color="blue", label="Serie procesada")

    ax[0].set_title(
        f"Area pupilar en funcion del tiempo ({METODO_VIDEO} + {OPERACION_VIDEO}) - {nombre}"
    )
    ax[0].set_ylabel("Area (px²)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, diam_o, color="gray", alpha=0.5)
    ax[1].plot(t, diam_p, color="blue")

    ax[1].set_title("Diametro pupilar en funcion del tiempo")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Diametro equivalente (px)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    cerrar_figura(fig, f"area_diametro_{nombre}", mostrar)


def graficar_luz(out, eventos_luz, nombre, mostrar):
    """3.2.3: deteccion automatica del estimulo luminoso."""

    t = out["tiempo_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(t, out["luminosidad_suave"], color="orange", label="Luminosidad")

    usadas = set()

    for t_ev, signo, _ in eventos_luz:

        etiqueta = "Encendido estimado" if signo > 0 else "Apagado estimado"

        ax.axvline(
            t_ev,
            color="red" if signo > 0 else "blue",
            linestyle="--",
            label=etiqueta if etiqueta not in usadas else None
        )

        usadas.add(etiqueta)

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Intensidad mediana (0-255)")
    ax.set_title(f"Deteccion automatica del estimulo luminoso - {nombre}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    cerrar_figura(fig, f"estimulo_{nombre}", mostrar)


def graficar_parametros(out, P, nombre, mostrar):
    """3.2.3: parametros sobre la curva de area pupilar."""

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        out["tiempo_s"], out["area_procesada_px2"],
        color="blue", label="Area pupilar procesada"
    )

    if np.isfinite(P["tiempo_estimulo_s"]):
        ax.axvline(P["tiempo_estimulo_s"], color="red",
                   linestyle="--", label="Estimulo")

    if np.isfinite(P["tiempo_apagado_s"]):
        ax.axvline(P["tiempo_apagado_s"], color="navy",
                   linestyle="--", label="Apagado")

    if np.isfinite(P["tiempo_inicio_s"]):
        ax.axvline(P["tiempo_inicio_s"], color="green",
                   linestyle="--", label="Inicio de constriccion")

    if np.isfinite(P["tiempo_minimo_s"]):
        ax.scatter(P["tiempo_minimo_s"], P["area_minima_px2"],
                   color="black", s=50, label="Area minima")

    if np.isfinite(P["area_basal_px2"]):
        ax.axhline(P["area_basal_px2"], color="gray",
                   linestyle=":", label="Area basal")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Area pupilar (px²)")
    ax.set_title(f"Respuesta pupilar al estimulo luminoso - {nombre}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    cerrar_figura(fig, f"parametros_{nombre}", mostrar)


def estado_texto(P):
    c, r = P["estado_constriccion"], P["estado_redilatacion"]

    partes = []

    if not c.startswith("No aplica"):
        partes.append(f"Constriccion: {c}")

    if not r.startswith("No aplica"):
        partes.append(f"Redilatacion: {r}")

    return " | ".join(partes) if partes else "Sin transiciones de luz detectadas"


def fila_resumen(nombre_archivo, P):
    """Fila corta y redondeada: es el formato fijo de salida."""

    def r(valor, dec):
        return round(float(valor), dec) if np.isfinite(valor) else np.nan

    return {
        "video": nombre_archivo,
        "tipo_registro": P["tipo_registro"],
        "encendido_s": r(P["tiempo_estimulo_s"], 3),
        "apagado_s": r(P["tiempo_apagado_s"], 3),
        "area_basal_px2": r(P["area_basal_px2"], 1),
        "area_minima_px2": r(P["area_minima_px2"], 1),
        "latencia_s": r(P["latencia_s"], 3),
        "amplitud_px2": r(P["amplitud_px2"], 1),
        "amplitud_pct": r(P["amplitud_pct"], 1),
        "vel_constriccion_px2_s": r(P["velocidad_constriccion_px2_s"], 1),
        "vel_redilatacion_px2_s": r(P["velocidad_redilatacion_px2_s"], 1),
        "estado": estado_texto(P)
    }


def imprimir_resumen_registro(fila):

    campos = [
        ("tipo_registro", "Tipo de registro", ""),
        ("encendido_s", "Encendido de la luz", "s"),
        ("apagado_s", "Apagado de la luz", "s"),
        ("latencia_s", "Latencia", "s"),
        ("amplitud_px2", "Amplitud de constriccion", "px²"),
        ("amplitud_pct", "Amplitud de constriccion", "% del basal"),
        ("vel_constriccion_px2_s", "Velocidad de constriccion", "px²/s"),
        ("vel_redilatacion_px2_s", "Velocidad de redilatacion", "px²/s"),
        ("estado", "Estado", "")
    ]

    filas = []

    for clave, nombre_campo, unidad in campos:

        valor = fila[clave]

        if isinstance(valor, float) and not np.isfinite(valor):
            valor = "-"

        filas.append({"Parametro": nombre_campo, "Valor": valor, "Unidad": unidad})

    print(pd.DataFrame(filas).to_string(index=False))


def tabla_variabilidad(resumen):
    """Media ± desvio entre registros, separado por tipo A / B.
    Las latencias marcadas para revision (< 0.1 s) no entran en el promedio."""

    parametros = [
        ("latencia_s", "Latencia (s)", 3),
        ("amplitud_px2", "Amplitud (px²)", 1),
        ("amplitud_pct", "Amplitud (% del basal)", 1),
        ("vel_constriccion_px2_s", "Vel. constriccion (px²/s)", 1),
        ("vel_redilatacion_px2_s", "Vel. redilatacion (px²/s)", 1)
    ]

    filas = []

    for clave, etiqueta, dec in parametros:

        fila = {"Parametro": etiqueta}

        for tipo in ("A", "B"):

            seleccion = resumen["tipo_registro"] == tipo

            if clave == "latencia_s":
                seleccion = seleccion & ~resumen["estado"].astype(str).str.contains("revisar")

            v = pd.to_numeric(resumen.loc[seleccion, clave], errors="coerce").dropna()

            if len(v) == 0:
                fila[f"Registro {tipo}"] = "-"
            elif len(v) == 1:
                fila[f"Registro {tipo}"] = f"{v.iloc[0]:.{dec}f} (n=1)"
            else:
                fila[f"Registro {tipo}"] = (
                    f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f} (n={len(v)})"
                )

        filas.append(fila)

    return pd.DataFrame(filas)

# ------------------------------------------------------------
# 3.1 DESCRIPTORES MORFOMETRICOS Y DE TEXTURA (frame segmentado)
# ------------------------------------------------------------

def analizar_frame_segmentado(roi, roi_gris, mascara_final, mejor_metodo, mejor_operacion):
    """3.1: descriptores de forma y texturas (1er y 2do orden) de la pupila
    y de un anillo aproximado de iris, sobre la ROI segmentada del Modulo 2."""

    # MODULO 3: Extraccion de caracteristicas y dinamica temporal

    # 3.1 Descriptores morfometricos y de textura

    # Utilizamos la mascara final seleccionada en el Modulo 2
    # (mejor metodo de segmentacion + mejor operacion morfologica).

    # La mascara indica que pixeles pertenecen a la pupila.
    # Para analizar intensidades y texturas usamos la ROI original.

    if mascara_final is None:
        raise ValueError("Se requiere la mascara final del Modulo 2")

    # CAMBIO: nombre dinamico segun el metodo/operacion que realmente ganaron
    NOMBRE_MASCARA_FINAL = f"{mejor_metodo} + {mejor_operacion}"

    mascara_pupila = mascara_final.copy()
    mascara_pupila = (mascara_pupila > 0).astype(np.uint8) * 255

    if mascara_pupila.shape != roi_gris.shape:
        raise ValueError(
            "La mascara final y la ROI tienen dimensiones diferentes"
        )

    if np.count_nonzero(mascara_pupila) == 0:
        raise ValueError("La mascara pupilar esta vacia")

    # Convertir la ROI original a RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    print(f"Mascara utilizada: {NOMBRE_MASCARA_FINAL}")


    # 3.1.1 Descriptores morfometricos

    # Obtener contornos de la mascara
    contornos, _ = cv2.findContours(
        mascara_pupila,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contornos:
        raise ValueError("No se encontraron contornos en la mascara")

    # Seleccionar el contorno de mayor area
    contorno_pupila = max(contornos, key=cv2.contourArea)

    # Crear mascara con la region principal
    mascara_pupila_3 = np.zeros_like(mascara_pupila)

    cv2.drawContours(
        mascara_pupila_3,
        [contorno_pupila],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Area
    area_pupila = np.count_nonzero(mascara_pupila_3)

    # Perimetro
    perimetro_pupila = cv2.arcLength(
        contorno_pupila,
        True
    )

    # Centroide
    momentos = cv2.moments(
        mascara_pupila_3,
        binaryImage=True
    )

    if momentos["m00"] == 0:
        raise ValueError("No se pudo calcular el centroide")

    centroide_x = momentos["m10"] / momentos["m00"]
    centroide_y = momentos["m01"] / momentos["m00"]

    # Circularidad
    circularidad = (
        4 * np.pi * area_pupila / perimetro_pupila**2
        if perimetro_pupila > 0 else np.nan
    )

    # Diametro equivalente
    diametro_equivalente = 2 * np.sqrt(
        area_pupila / np.pi
    )

    # Excentricidad mediante ajuste de elipse
    if len(contorno_pupila) >= 5:

        elipse = cv2.fitEllipse(contorno_pupila)

        eje_1, eje_2 = elipse[1]

        eje_mayor = max(eje_1, eje_2)
        eje_menor = min(eje_1, eje_2)

        excentricidad = (
            np.sqrt(max(0, 1 - (eje_menor / eje_mayor)**2))
            if eje_mayor > 0 else np.nan
        )

    else:
        excentricidad = np.nan


    print("\n--- 3.1.1 Descriptores morfometricos ---")
    print(f"Area: {area_pupila} pixeles cuadrados")
    print(f"Perimetro: {perimetro_pupila:.2f} pixeles")
    print(f"Centroide: ({centroide_x:.2f}, {centroide_y:.2f})")
    print(f"Circularidad: {circularidad:.4f}")
    print(f"Excentricidad: {excentricidad:.4f}")
    print(f"Diametro equivalente: {diametro_equivalente:.2f} pixeles")


    # Visualizar mascara y descriptores geometricos
    roi_morfometria = roi_rgb.copy()

    # Contorno en verde
    cv2.drawContours(
        roi_morfometria,
        [contorno_pupila],
        -1,
        (0, 255, 0),
        1
    )

    # Centroide en rojo
    cv2.circle(
        roi_morfometria,
        (int(round(centroide_x)), int(round(centroide_y))),
        2,
        (255, 0, 0),
        -1
    )

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))

    ax[0].imshow(roi_rgb)
    ax[0].set_title("ROI original")

    ax[1].imshow(
        mascara_pupila_3,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    # CAMBIO: titulo dinamico (antes decia "Watershed + dilatacion" fijo)
    ax[1].set_title(NOMBRE_MASCARA_FINAL)

    ax[2].imshow(roi_morfometria)
    ax[2].set_title("Contorno y centroide")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()



    # 3.1.2 Textura de primer orden para la pupila

    # Analizar intensidades RGB dentro de la mascara pupilar

    canales_rgb = cv2.split(roi_rgb)
    nombres_canales = ["R", "G", "B"]

    seleccion_pupila = mascara_pupila_3 > 0

    estadisticas_primer_orden = {}

    for nombre, canal in zip(nombres_canales, canales_rgb):

        pixeles = canal[seleccion_pupila]

        # Media y desvio estandar
        media = np.mean(pixeles)
        desvio = np.std(pixeles)

        # Histograma normalizado
        histograma = np.bincount(
            pixeles,
            minlength=256
        )

        probabilidades = histograma / histograma.sum()
        p_no_cero = probabilidades[probabilidades > 0]

        # Entropia de Shannon
        entropia = -np.sum(
            p_no_cero * np.log2(p_no_cero)
        )

        # Asimetria
        asimetria = skew(pixeles) if desvio > 0 else 0.0

        estadisticas_primer_orden[nombre] = {
            "Media": media,
            "Desvio estandar": desvio,
            "Entropia": entropia,
            "Asimetria": asimetria
        }


    # Tabla de primer orden de la pupila
    tabla_primer_orden_pupila = pd.DataFrame.from_dict(
        estadisticas_primer_orden,
        orient="index"
    )

    tabla_primer_orden_pupila.index.name = "Canal RGB"

    print("\n--- 3.1.2 Textura de primer orden: pupila ---")
    display(tabla_primer_orden_pupila.round(4))


    # Histogramas RGB de la pupila
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    for i, (nombre, canal) in enumerate(
        zip(nombres_canales, canales_rgb)
    ):

        ax[i].hist(
            canal[seleccion_pupila],
            bins=32,
            range=(0, 256)
        )

        ax[i].set_title(f"Canal {nombre}")
        ax[i].set_xlabel("Intensidad")
        ax[i].set_ylabel("Cantidad de pixeles")

    plt.tight_layout()
    plt.show()



    # 3.1.3 Textura de segundo orden: GLCM

    # Utilizamos:
    # - 32 niveles de gris
    # - distancia de 1 pixel
    # - orientaciones 0, 45, 90 y 135 grados
    # - GLCM simetrica y normalizada
    #
    # Solo contamos pares de pixeles dentro de la mascara.

    niveles_glcm = 32
    distancias_glcm = [1]

    angulos_glcm = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]


    # Funcion para construir una GLCM dentro de una mascara
    def glcm_con_mascara(imagen, mascara, niveles, distancias, angulos):

        matriz = np.zeros(
            (
                niveles,
                niveles,
                len(distancias),
                len(angulos)
            ),
            dtype=np.float64
        )

        alto, ancho = imagen.shape

        for i_d, distancia in enumerate(distancias):

            for i_a, angulo in enumerate(angulos):

                # Desplazamiento de los pixeles vecinos
                dx = int(round(distancia * np.cos(angulo)))
                dy = int(round(-distancia * np.sin(angulo)))

                # Limites validos
                x_inicio = max(0, -dx)
                x_fin = min(ancho, ancho - dx)

                y_inicio = max(0, -dy)
                y_fin = min(alto, alto - dy)

                referencia = imagen[
                    y_inicio:y_fin,
                    x_inicio:x_fin
                ]

                vecina = imagen[
                    y_inicio + dy:y_fin + dy,
                    x_inicio + dx:x_fin + dx
                ]

                mascara_ref = mascara[
                    y_inicio:y_fin,
                    x_inicio:x_fin
                ]

                mascara_vecina = mascara[
                    y_inicio + dy:y_fin + dy,
                    x_inicio + dx:x_fin + dx
                ]

                # Ambos pixeles deben pertenecer a la region
                validos = (
                    (mascara_ref > 0) &
                    (mascara_vecina > 0)
                )

                valores_ref = referencia[validos]
                valores_vecinos = vecina[validos]

                # Contar ocurrencias de cada par
                np.add.at(
                    matriz[:, :, i_d, i_a],
                    (valores_ref, valores_vecinos),
                    1
                )

                # Matriz simetrica
                matriz[:, :, i_d, i_a] += (
                    matriz[:, :, i_d, i_a].T.copy()
                )

                # Normalizar a probabilidades
                total = matriz[:, :, i_d, i_a].sum()

                if total > 0:
                    matriz[:, :, i_d, i_a] /= total

        return matriz


    # Cuantizar la ROI de 256 a 32 niveles
    imagen_cuantizada = (
        roi_gris // (256 // niveles_glcm)
    ).astype(np.uint8)

    # Construir GLCM de la pupila
    glcm_pupila = glcm_con_mascara(
        imagen_cuantizada,
        mascara_pupila_3,
        niveles_glcm,
        distancias_glcm,
        angulos_glcm
    )

    if np.any(glcm_pupila.sum(axis=(0, 1)) == 0):
        raise ValueError(
            "No hay suficientes pares de pixeles para calcular la GLCM"
        )


    # Calcular atributos GLCM
    contraste_glcm = graycoprops(
        glcm_pupila,
        "contrast"
    )

    homogeneidad_glcm = graycoprops(
        glcm_pupila,
        "homogeneity"
    )

    correlacion_glcm = graycoprops(
        glcm_pupila,
        "correlation"
    )

    # Energia: suma de cuadrados de probabilidades
    energia_glcm = np.sum(
        glcm_pupila**2,
        axis=(0, 1)
    )

    # Entropia de cada orientacion
    entropia_glcm = np.zeros(
        (len(distancias_glcm), len(angulos_glcm))
    )

    for i_d in range(len(distancias_glcm)):

        for i_a in range(len(angulos_glcm)):

            P = glcm_pupila[:, :, i_d, i_a]
            p_no_cero = P[P > 0]

            entropia_glcm[i_d, i_a] = -np.sum(
                p_no_cero * np.log2(p_no_cero)
            )


    # TABLA GLCM DE LA PUPILA
    # Reemplaza los print individuales para cada angulo.

    angulos_grados = [0, 45, 90, 135]

    resultados_glcm_pupila = []

    for i_d, distancia in enumerate(distancias_glcm):

        for i_a, angulo in enumerate(angulos_grados):

            resultados_glcm_pupila.append({
                "Distancia (px)": distancia,
                "Angulo (grados)": angulo,
                "Contraste": contraste_glcm[i_d, i_a],
                "Energia": energia_glcm[i_d, i_a],
                "Homogeneidad": homogeneidad_glcm[i_d, i_a],
                "Correlacion": correlacion_glcm[i_d, i_a],
                "Entropia": entropia_glcm[i_d, i_a]
            })

    tabla_glcm_pupila = pd.DataFrame(resultados_glcm_pupila)

    print("\n--- 3.1.3 Textura de segundo orden: GLCM pupilar ---")
    display(tabla_glcm_pupila.round(4))


    # Visualizacion de la GLCM a 0 grados
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(
        roi_gris,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    ax[0].set_title("ROI original")

    ax[1].imshow(
        imagen_cuantizada,
        cmap="gray",
        vmin=0,
        vmax=niveles_glcm - 1
    )
    ax[1].set_title("ROI cuantizada: 32 niveles")

    im = ax[2].imshow(
        glcm_pupila[:, :, 0, 0],
        cmap="Reds"
    )

    ax[2].set_title("GLCM pupilar: 0 grados")
    ax[2].set_xlabel("Nivel de gris vecino")
    ax[2].set_ylabel("Nivel de gris referencia")

    fig.colorbar(
        im,
        ax=ax[2],
        label="Probabilidad"
    )

    ax[0].axis("off")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


    # 3.1.4 Analisis de textura del iris

    # La mascara final corresponde a la pupila, no al iris.
    # Se define una region anular alrededor de la pupila
    # para analizar aproximadamente la textura del iris.

    radio_pupila = diametro_equivalente / 2

    alto_roi, ancho_roi = roi_gris.shape

    Y_iris, X_iris = np.ogrid[:alto_roi, :ancho_roi]

    distancia_iris = np.sqrt(
        (X_iris - centroide_x)**2 +
        (Y_iris - centroide_y)**2
    )

    # Radios de la region anular
    radio_interno = 1.15 * radio_pupila
    radio_externo = 1.65 * radio_pupila

    mascara_iris = (
        (distancia_iris >= radio_interno) &
        (distancia_iris <= radio_externo)
    ).astype(np.uint8) * 255

    if np.count_nonzero(mascara_iris) == 0:
        raise ValueError("La mascara aproximada del iris esta vacia")


    # Visualizar la region seleccionada

    iris_segmentado = cv2.bitwise_and(
        roi_rgb,
        roi_rgb,
        mask=mascara_iris
    )

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))

    ax[0].imshow(roi_rgb)
    ax[0].set_title("ROI original")

    ax[1].imshow(
        mascara_iris,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    ax[1].set_title("Mascara aproximada del iris")

    ax[2].imshow(iris_segmentado)
    ax[2].set_title("Region seleccionada")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


    # Textura de primer orden del iris en RGB

    canales_iris = cv2.split(roi_rgb)
    estadisticas_iris = {}

    for nombre, canal in zip(nombres_canales, canales_iris):

        pixeles = canal[mascara_iris > 0]

        media = np.mean(pixeles)
        desvio = np.std(pixeles)

        histograma = np.bincount(
            pixeles,
            minlength=256
        )

        probabilidades = histograma / histograma.sum()
        p_no_cero = probabilidades[probabilidades > 0]

        entropia = -np.sum(
            p_no_cero * np.log2(p_no_cero)
        )

        asimetria = (
            skew(pixeles)
            if desvio > 0 else 0.0
        )

        estadisticas_iris[nombre] = {
            "Media": media,
            "Desvio estandar": desvio,
            "Entropia": entropia,
            "Asimetria": asimetria
        }


    # Tabla de primer orden del iris

    tabla_primer_orden = pd.DataFrame.from_dict(
        estadisticas_iris,
        orient="index"
    )

    tabla_primer_orden.index.name = "Canal RGB"

    print("\n--- 3.1.4 Textura de primer orden del iris ---")
    display(tabla_primer_orden.round(4))


    # Textura de segundo orden del iris por canal RGB

    resultados_glcm_iris = {}

    for nombre, canal in zip(nombres_canales, canales_iris):

        # Cuantizacion a 32 niveles
        canal_cuantizado = (
            canal // (256 // niveles_glcm)
        ).astype(np.uint8)

        # GLCM dentro de la mascara del iris
        glcm_canal = glcm_con_mascara(
            canal_cuantizado,
            mascara_iris,
            niveles_glcm,
            distancias_glcm,
            angulos_glcm
        )

        if np.any(glcm_canal.sum(axis=(0, 1)) == 0):
            raise ValueError(
                f"No hay suficientes pares de pixeles en el canal {nombre}"
            )

        # Atributos GLCM
        contraste = graycoprops(
            glcm_canal,
            "contrast"
        )

        homogeneidad = graycoprops(
            glcm_canal,
            "homogeneity"
        )

        correlacion = graycoprops(
            glcm_canal,
            "correlation"
        )

        energia = np.sum(
            glcm_canal**2,
            axis=(0, 1)
        )

        # Entropia promediada sobre distancias y orientaciones
        entropias = []

        for i_d in range(len(distancias_glcm)):

            for i_a in range(len(angulos_glcm)):

                P = glcm_canal[:, :, i_d, i_a]
                p_no_cero = P[P > 0]

                entropias.append(
                    -np.sum(p_no_cero * np.log2(p_no_cero))
                )

        resultados_glcm_iris[nombre] = {
            "Contraste": np.mean(contraste),
            "Energia": np.mean(energia),
            "Homogeneidad": np.mean(homogeneidad),
            "Correlacion": np.mean(correlacion),
            "Entropia": np.mean(entropias)
        }


    # Tabla de segundo orden del iris

    tabla_segundo_orden = pd.DataFrame.from_dict(
        resultados_glcm_iris,
        orient="index"
    )

    tabla_segundo_orden.index.name = "Canal RGB"

    print("\n--- 3.1.4 Textura de segundo orden del iris (GLCM) ---")
    display(tabla_segundo_orden.round(4))


    # Comparacion grafica del contraste GLCM entre canales RGB

    plt.figure(figsize=(6, 4))

    plt.bar(
        nombres_canales,
        tabla_segundo_orden.loc[nombres_canales, "Contraste"]
    )

    plt.title("Contraste GLCM del iris por canal RGB")
    plt.xlabel("Canal RGB")
    plt.ylabel("Contraste promedio")

    plt.tight_layout()
    plt.show()

    # 3.1.3 / 3.1.4 Conclusion: que canal aporta mayor riqueza textural

    print("\n--- Conclusion de textura: comparacion entre canales RGB ---")

    metricas_conclusion = tabla_segundo_orden[
        ["Contraste", "Entropia", "Energia", "Homogeneidad"]
    ].copy()

    metricas_conclusion["Desvio (1er orden)"] = (
        tabla_primer_orden["Desvio estandar"]
    )

    display(metricas_conclusion.round(4))

    # Criterio: MAS textura = mas contraste, mas entropia, mas desvio,
    # y MENOS homogeneidad y energia (la imagen es menos uniforme).
    puntaje_textura = (
        metricas_conclusion["Contraste"].rank()
        + metricas_conclusion["Entropia"].rank()
        + metricas_conclusion["Desvio (1er orden)"].rank()
        + (-metricas_conclusion["Homogeneidad"]).rank()
        + (-metricas_conclusion["Energia"]).rank()
    )

    canal_mas_rico = puntaje_textura.idxmax()

    print(f"Mayor contraste:         canal {metricas_conclusion['Contraste'].idxmax()}")
    print(f"Mayor entropia (GLCM):   canal {metricas_conclusion['Entropia'].idxmax()}")
    print(f"Mayor desvio (1er ord.): canal {metricas_conclusion['Desvio (1er orden)'].idxmax()}")
    print(f"Menor homogeneidad:      canal {metricas_conclusion['Homogeneidad'].idxmin()}")
    print(f"Menor energia:           canal {metricas_conclusion['Energia'].idxmin()}")
    print(f"\n=> Canal con mayor riqueza textural (ranking combinado): {canal_mas_rico}")
    print(
        "Nota: la 'region iris' es un anillo aproximado "
        "(1.15 a 1.65 radios pupilares alrededor del centroide), "
        "no un iris segmentado. Puede incluir esclerotica, parpado o "
        "pestanas, asi que la conclusion es indicativa y depende del "
        "color de ojo y de la iluminacion del registro."
    )

    # Grafico comparativo de las cuatro metricas por canal
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))

    for a, metrica in zip(
        ax, ["Contraste", "Entropia", "Homogeneidad", "Energia"]
    ):
        a.bar(
            nombres_canales,
            metricas_conclusion.loc[nombres_canales, metrica],
            color=["tab:red", "tab:green", "tab:blue"]
        )
        a.set_title(metrica)
        a.set_xlabel("Canal RGB")

    plt.suptitle("Textura GLCM del iris (anillo aproximado) por canal")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3.2 PROCESAMIENTO EN VIDEO (de punta a punta, sin intervencion)
# ------------------------------------------------------------

def procesar_registros(video_path, carpeta_resultados, metodo, operacion,
                       modo="desarrollo", procesar_otros=None, mostrar_figuras=None):
    """
    Procesa video_path: curva de area/diametro frame a frame, parametros
    fisiologicos, tablas y graficos. En modo "desarrollo" tambien procesa
    los otros videos de la misma carpeta (variabilidad entre registros).

    Devuelve (resumen, variabilidad): dos DataFrames.
    """

    video_path = Path(video_path)
    carpeta_resultados = Path(carpeta_resultados)

    if not video_path.is_file():
        raise FileNotFoundError(f"No se encontro el video: {video_path}")

    if procesar_otros is None:
        procesar_otros = (modo == "desarrollo")

    if mostrar_figuras is None:
        mostrar_figuras = (modo == "desarrollo")

    configurar(metodo, operacion, carpeta_resultados, mostrar_figuras)

    print(f"Metodo de segmentacion para el video: {metodo}")
    print(f"Operacion morfologica para el video: {operacion}")

    semilla_principal = estimar_semilla(video_path)

    print("\n--- Semilla de la pupila ---")
    print(f"Centro: ({semilla_principal['cx']:.1f}, {semilla_principal['cy']:.1f}) px")
    print(f"Radio: {semilla_principal['radio']:.1f} px")
    print(
        f"Frames consistentes: {semilla_principal['n_validas']} "
        f"de {semilla_principal['n_muestras']}"
    )


    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    CARPETA_DETALLE.mkdir(parents=True, exist_ok=True)

    principal = video_path.resolve()

    lista_videos = [video_path]

    if procesar_otros:
        lista_videos += sorted(
            p for p in video_path.parent.iterdir()
            if p.suffix.lower() in EXTENSIONES_VIDEO
            and p.resolve() != principal
        )

    clave_metodo = re.sub(
        r"[^A-Za-z0-9_=+-]", "_", f"{METODO_VIDEO}__{OPERACION_VIDEO}"
    )

    filas_resumen = []

    for ruta in lista_videos:

        es_principal = ruta.resolve() == principal
        mostrar = MOSTRAR_FIGURAS and es_principal
        nombre = ruta.stem
        t_inicio = time.time()

        print(f"\n{'=' * 60}\nRegistro: {ruta.name}"
              f"{'  (principal)' if es_principal else ''}\n{'=' * 60}")

        try:

            ruta_mediciones = CARPETA_DETALLE / f"mediciones_{nombre}__{clave_metodo}.csv"
            revision = {}

            if (not es_principal) and USAR_CACHE_OTROS and ruta_mediciones.exists():

                df_crudo = pd.read_csv(ruta_mediciones)
                fps_reg = (len(df_crudo) - 1) / df_crudo["tiempo_s"].iloc[-1]
                print("  Mediciones cargadas de cache (carpeta detalle).")

            else:

                df_crudo, fps_reg, revision, _ = procesar_video(
                    ruta,
                    semilla=semilla_principal if es_principal else None,
                    guardar_revision=(es_principal and modo == "desarrollo")
                )

                df_crudo.to_csv(ruta_mediciones, index=False)

                if es_principal and modo == "desarrollo":
                    print("  Motivos de descarte de intentos de deteccion:")
                    for motivo, cantidad in MOTIVOS_RECHAZO.most_common():
                        print(f"    {motivo}: {cantidad}")

            out, P, eventos_luz, conteo = analizar_registro(df_crudo, fps_reg)

            pct_deteccion = 100 * out["detectada"].mean()

            print(f"  {len(out)} frames | {out['tiempo_s'].iloc[-1]:.1f} s | "
                  f"detecciones validas: {pct_deteccion:.1f} %")
            print(f"  Artefactos: {conteo['n_parpadeos']} parpadeos, "
                  f"{conteo['n_perdidas_seguimiento']} perdidas de seguimiento, "
                  f"{conteo['n_mov_oculares']} saltos oculares, "
                  f"{conteo['n_descartados']} mediciones descartadas\n")

            resumen_fila = fila_resumen(ruta.name, P)
            filas_resumen.append(resumen_fila)

            imprimir_resumen_registro(resumen_fila)

            # --- Archivos de salida claros (formato fijo) ---
            pd.DataFrame([resumen_fila]).to_csv(
                carpeta_resultados / f"parametros_{nombre}.csv",
                index=False, encoding="utf-8-sig"
            )

            serie_clara = pd.DataFrame({
                "frame": out["frame"],
                "tiempo_s": out["tiempo_s"].round(3),
                "area_px2": out["area_procesada_px2"].round(1),
                "diametro_px": out["diametro_procesado_px"].round(2),
                "evento": out["evento"]
            })

            serie_clara.to_csv(
                carpeta_resultados / f"serie_{nombre}.csv", index=False
            )

            # --- Detalle completo (para validacion y revision) ---
            out.round(4).to_csv(
                CARPETA_DETALLE / f"serie_detallada_{nombre}.csv", index=False
            )

            pd.DataFrame([{
                "video": ruta.name, "metodo": METODO_VIDEO,
                "morfologia": OPERACION_VIDEO,
                "fps": fps_reg, "duracion_s": out["tiempo_s"].iloc[-1],
                "porcentaje_deteccion": pct_deteccion,
                **conteo, **P
            }]).round(4).to_csv(
                CARPETA_DETALLE / f"parametros_detallados_{nombre}.csv", index=False
            )

            # --- Archivos con los nombres y columnas del formato anterior ---
            if GUARDAR_FORMATO_ANTERIOR:

                df_crudo[[
                    "frame", "tiempo_s", "area_px2", "diametro_px", "centro_x",
                    "centro_y", "circularidad", "puntaje_deteccion", "detectada"
                ]].to_csv(carpeta_resultados / f"mediciones_{nombre}.csv", index=False)

                out[[
                    "tiempo_s", "area_original_px2", "area_procesada_px2",
                    "diametro_original_px", "diametro_procesado_px"
                ]].to_csv(carpeta_resultados / f"serie_procesada_{nombre}.csv", index=False)

                pd.DataFrame([{
                    "video": ruta.name,
                    "metodo": metodo,
                    "morfologia": operacion,
                    "tiempo_estimulo_s": P["tiempo_estimulo_s"],
                    "tiempo_apagado_s": P["tiempo_apagado_s"],
                    "area_basal_px2": P["area_basal_px2"],
                    "area_minima_px2": P["area_minima_px2"],
                    "tiempo_inicio_s": P["tiempo_inicio_s"],
                    "tiempo_minimo_s": P["tiempo_minimo_s"],
                    "latencia_s": P["latencia_s"],
                    "amplitud_px2": P["amplitud_px2"],
                    "velocidad_constriccion_px2_s": P["velocidad_constriccion_px2_s"],
                    "velocidad_redilatacion_px2_s": P["velocidad_redilatacion_px2_s"],
                    "estado": estado_texto(P)
                }]).to_csv(
                    carpeta_resultados / f"parametros_fisiologicos_{nombre}.csv",
                    index=False
                )

            # --- Graficos ---
            graficar_curvas(out, nombre, mostrar)
            graficar_luz(out, eventos_luz, nombre, mostrar)
            graficar_parametros(out, P, nombre, mostrar)

            if es_principal and revision:
                mostrar_revision(revision, fps_reg, nombre, mostrar)

            print(f"\n  Tiempo de ejecucion: {time.time() - t_inicio:.1f} s")

        except Exception as error:

            if es_principal:
                raise

            print(f"  [!] Se omite este video: {error}")

            filas_resumen.append({
                "video": ruta.name, "tipo_registro": "Error",
                "estado": str(error)
            })


    # ------------------------------------------------------------
    # TABLAS FINALES
    # ------------------------------------------------------------

    resumen = pd.DataFrame(filas_resumen)

    resumen.to_csv(
        carpeta_resultados / "resumen_registros.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\n{'=' * 60}\nRESUMEN DE REGISTROS\n{'=' * 60}")

    columnas_pantalla = [
        "video", "tipo_registro", "latencia_s", "amplitud_px2",
        "amplitud_pct", "vel_constriccion_px2_s", "vel_redilatacion_px2_s"
    ]

    print(
        resumen[[c for c in columnas_pantalla if c in resumen.columns]]
        .fillna("-").to_string(index=False)
    )

    variabilidad = tabla_variabilidad(resumen)

    variabilidad.to_csv(
        carpeta_resultados / "variabilidad_entre_registros.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\n{'=' * 60}\nVARIABILIDAD ENTRE REGISTROS (media ± desvio)\n{'=' * 60}")
    print(variabilidad.to_string(index=False))

    # Excel con las dos tablas (opcional: necesita openpyxl)
    try:
        with pd.ExcelWriter(carpeta_resultados / "resumen_resultados.xlsx") as escritor:
            resumen.to_excel(escritor, sheet_name="Registros", index=False)
            variabilidad.to_excel(escritor, sheet_name="Variabilidad", index=False)
        print("\nExcel generado: resultados/resumen_resultados.xlsx")
    except Exception:
        print("\n(Para generar tambien el Excel: pip install openpyxl)")

    print("\n--- 3.2.4 Metodo de filtrado/interpolacion de artefactos ---")
    print("1. Frames sin deteccion o rechazados por controles de calidad -> NaN.")
    print("2. Filtro de Hampel (ventana 7, 3 sigmas robustas, piso 5%) -> atipicos a NaN.")
    print("3. Filtro de mediana local (0.3 s, tolerancia 35%) -> saltos de varios frames a NaN.")
    print("4. Margen de 0.05 s alrededor de cada parpadeo y descarte de tramos validos < 0.1 s.")
    print("5. Interpolacion lineal solo en huecos de hasta 3 frames; mediana de 5 frames.")
    print("6. Columna 'evento': parpadeo / perdida_seguimiento / mov_ocular.")
    print(f"\nResultados en: {carpeta_resultados}")

    return resumen, variabilidad



def modulo3(frame, roi, roi_gris, mascara_final, mejor_metodo, mejor_operacion,
            VIDEO_PATH, CARPETA_PROYECTO):
    """Modulo 3 completo para el modo "desarrollo": 3.1 + 3.2.

    Los resultados del Modulo 2 llegan como parametros; VIDEO_PATH se comparte
    con el resto del proyecto para analizar el mismo video.
    """
    VIDEO_PATH = Path(VIDEO_PATH)
    CARPETA_PROYECTO = Path(CARPETA_PROYECTO)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el video: {VIDEO_PATH}")

    if mascara_final is None or mascara_final.shape != roi_gris.shape:
        raise ValueError("La mascara del Modulo 2 no coincide con la ROI")

    analizar_frame_segmentado(
        roi, roi_gris, mascara_final, mejor_metodo, mejor_operacion
    )

    return procesar_registros(
        VIDEO_PATH,
        CARPETA_PROYECTO / "resultados",
        mejor_metodo,
        mejor_operacion,
        modo="desarrollo",
        procesar_otros=PROCESAR_TODOS_LOS_VIDEOS
    )