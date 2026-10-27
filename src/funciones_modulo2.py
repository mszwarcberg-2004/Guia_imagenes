#Este archivo contiene funciones específicas correspondientes al
#módulo 2

#Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour


# 2.1.2 Segmentacion mediante K-Means

def segmentar_kmeans(imagen_kmeans, k):
    # Convertir la imagen en una lista de pixeles
    pixeles = imagen_kmeans.reshape(-1, 1).astype(np.float32)

    # Criterio de finalizacion del algoritmo
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

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

    return imagen_agrupada, mascara, centros


# 2.1.3 Segmentacion mediante contornos activos (Snakes)

def segmentar_snake(roi_gris):
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
        alpha=0.025,
        beta=10,
        gamma=0.01,
        w_line=0,
        w_edge=5,
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

    return mascara_snake, contorno_inicial, contorno_final

# Visualizar el contorno inicial y final
def mostrar_snake(roi_gris, mascara_snake, contorno_inicial, contorno_final):
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

def segmentar_watershed(roi_gris, mascara_otsu):
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

    return mascara_inicial, mascara_watershed

# 2.1.5 Comparacion de los metodos mediante Dice

# A. Crear o cargar la mascara de referencia manual
def cargar_o_crear_referencia(REFERENCIA_PATH, roi_gris):

    if REFERENCIA_PATH.exists():

        mascara_referencia = cv2.imread(str(REFERENCIA_PATH), cv2.IMREAD_GRAYSCALE)

        if mascara_referencia.shape != roi_gris.shape:
            raise ValueError("La mascara de referencia no tiene las dimensiones de la ROI")

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
        mascara_referencia = np.zeros_like(roi_gris, dtype=np.uint8)

        cv2.fillPoly(mascara_referencia, [puntos], 255)

        # Guardar para reutilizar la misma referencia
        REFERENCIA_PATH.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(REFERENCIA_PATH), mascara_referencia)

    return mascara_referencia


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


# 2.2.1 Operaciones morfologicas

def aplicar_morfologia(mascara_base):
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

    # Mascaras que vamos a comparar
    return {
        "Sin morfologia": mascara_base,
        "Erosion": mascara_erosion,
        "Dilatacion": mascara_dilatacion,
        "Apertura": mascara_apertura,
        "Cierre": mascara_cierre
    }


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
