# Modulo 3 - Integracion del codigo de analisis de la pupila
# Adaptado del codigo de trabajo del equipo, sin modificar modulos 1 y 2.
# Pendiente validar experimentalmente las mediciones del video.

from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature import graycoprops
from skimage.filters import gaussian, sobel
from skimage.segmentation import active_contour, watershed


def modulo3(frame, roi, roi_gris, mascara_final, mejor_metodo, mejor_operacion, VIDEO_PATH, CARPETA_PROYECTO):
    """Ejecuta morfometria, texturas y seguimiento temporal del Modulo 3.

    Los resultados del modulo 2 llegan como parametros; VIDEO_PATH se comparte
    con el resto del proyecto para analizar el mismo video.
    """
    VIDEO_PATH = Path(VIDEO_PATH)
    CARPETA_PROYECTO = Path(CARPETA_PROYECTO)
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el video: {VIDEO_PATH}")
    if mascara_final is None or mascara_final.shape != roi_gris.shape:
        raise ValueError("La mascara del Modulo 2 no coincide con la ROI")

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
    print(tabla_primer_orden_pupila.round(4).to_string())


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
    print(tabla_glcm_pupila.round(4).to_string(index=False))


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
    print(tabla_primer_orden.round(4).to_string())


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
    print(tabla_segundo_orden.round(4).to_string())


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

    # MODULO 3.2: Procesamiento temporal de la pupila

    # 3.2.1 Automatizacion de la segmentacion en video

    # Estrategia:
    # 1. Estimar una "semilla" (centro y radio tipicos de la pupila) usando
    #    varios frames repartidos en todo el video y quedandose con el consenso.
    #    Sabemos que, como la cabeza esta fija, la pupila siempre esta cerca de la semilla.
    # 2. En cada frame, detectar un centro/radio candidato dentro de esa zona
    #    (oscuro, uniforme y rodeado de intensidades mas claras).
    # 3. Segmentar la pupila con el MEJOR METODO elegido en el Modulo 2
    #    (mejor_metodo, segun Dice contra la referencia manual).
    # 4. Refinar con la MEJOR OPERACION MORFOLOGICA del Modulo 2
    #    (mejor_operacion) y con la envolvente convexa (la pupila es convexa).
    # 5. Seguir el centro entre frames y descartar saltos imposibles.



    if not mejor_metodo or not mejor_operacion:
        raise ValueError("Se requiere el metodo y la morfologia elegidos en el Modulo 2")

    # Si se quiere forzar otro metodo manualmente, cambiar estas dos lineas
    # (opciones: "Otsu", "K-Means K=2".."K-Means K=5", "Snakes", "Watershed")
    METODO_VIDEO = mejor_metodo
    OPERACION_VIDEO = mejor_operacion

    print(f"Metodo de segmentacion para el video: {METODO_VIDEO}")
    print(f"Operacion morfologica para el video: {OPERACION_VIDEO}")

    carpeta_resultados = CARPETA_PROYECTO / "resultados"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    nombre_video = Path(VIDEO_PATH).stem


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


    # A. Detectar posibles centros pupilares

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


    # A.2 Estimar la semilla usando varios frames del video

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


    SEMILLA = estimar_semilla(VIDEO_PATH)

    print("\n--- Semilla de la pupila ---")
    print(f"Centro: ({SEMILLA['cx']:.1f}, {SEMILLA['cy']:.1f}) px")
    print(f"Radio: {SEMILLA['radio']:.1f} px")
    print(
        f"Frames consistentes: {SEMILLA['n_validas']} "
        f"de {SEMILLA['n_muestras']}"
    )

    # Zona de seguimiento: la cabeza esta fija, la pupila no se va lejos
    _m = FACTOR_ZONA_SEGUIMIENTO * SEMILLA["radio"]

    ZONA_SEGUIMIENTO = (
        max(0.0, SEMILLA["cx"] - _m),
        min(float(SEMILLA["ancho"]), SEMILLA["cx"] + _m),
        max(0.0, SEMILLA["cy"] - _m),
        min(float(SEMILLA["alto"]), SEMILLA["cy"] + _m)
    )

    _rmin = max(
        RADIO_MIN_REL * SEMILLA["ancho"],
        RELACION_RADIO_SEMILLA[0] * SEMILLA["radio"]
    )

    _rmax = min(
        RADIO_MAX_REL * SEMILLA["ancho"],
        RELACION_RADIO_SEMILLA[1] * SEMILLA["radio"]
    )

    if _rmin >= _rmax:
        _rmin, _rmax = 0.7 * SEMILLA["radio"], 1.4 * SEMILLA["radio"]

    RANGO_RADIOS_SEMILLA = (_rmin, _rmax)


    # B. Segmentar la pupila con el metodo elegido en el Modulo 2

    # Funciones locales (trabajan sobre la ROI recortada alrededor
    # del candidato), una por cada metodo del apartado 2.1.
    # Todas devuelven una mascara uint8 (0/255) del tamano de la ROI local,
    # o None si no pudieron segmentar.
    # zona_disco: booleana, limita el analisis a un entorno de la pupila.

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


    def aplicar_morfologia(mascara, operacion):

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


    # Registro de por que se descartan frames (para diagnosticar)
    from collections import Counter

    MOTIVOS_RECHAZO = Counter()


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
        mascara = aplicar_morfologia(mascara, OPERACION_VIDEO)

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


    # C. Procesar un frame con seguimiento temporal

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


    # D. Abrir y procesar video

    # Semilla fija para que K-Means sea reproducible
    cv2.setRNGSeed(0)

    video_3 = cv2.VideoCapture(str(VIDEO_PATH))

    if not video_3.isOpened():
        raise ValueError(f"No se pudo abrir {VIDEO_PATH}")

    fps_3 = video_3.get(cv2.CAP_PROP_FPS)


    if fps_3 <= 0:
        video_3.release()
        raise ValueError("FPS invalido")

    registros = []
    frames_revision = {}

    resultado_anterior = None
    fallos = 0
    indice_frame = 0

    total_frames = int(
        video_3.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Mostrar un frame cada 100 milisegundos
    FRECUENCIA_VISUALIZACION = 10  # Hz

    paso_frames = max(1, int(round(fps_3 / FRECUENCIA_VISUALIZACION)))

    indices_revision = set(
        range(0, total_frames, paso_frames)
    )

    while True:

        ret, frame = video_3.read()

        if not ret:
            break

        gris = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        resultado, candidato = procesar_frame_pupila(
            gris,
            resultado_anterior
        )

        if resultado is None:

            fallos += 1

            # Si se pierde el seguimiento, volver a buscar
            # en la zona de la semilla.
            if fallos >= 3:

                resultado_anterior = None

                resultado, candidato = procesar_frame_pupila(gris)

        if resultado is not None:

            resultado_anterior = resultado
            fallos = 0

            area = resultado["area"]
            diametro = resultado["diametro"]

            centro_x = resultado["cx"]
            centro_y = resultado["cy"]

            circularidad_frame = resultado["circularidad"]
            puntaje_frame = resultado["puntaje"]

            detectada = 1

        else:

            area = np.nan
            diametro = np.nan

            centro_x = np.nan
            centro_y = np.nan

            circularidad_frame = np.nan
            puntaje_frame = np.nan

            detectada = 0

        registros.append({
            "frame": indice_frame,
            "tiempo_s": indice_frame / fps_3,
            "area_px2": area,
            "diametro_px": diametro,
            "centro_x": centro_x,
            "centro_y": centro_y,
            "circularidad": circularidad_frame,
            "puntaje_deteccion": puntaje_frame,
            "detectada": detectada
        })

        if indice_frame in indices_revision:

            mascara = (
                resultado["mascara"].copy()
                if resultado is not None
                else np.zeros_like(gris)
            )

            frames_revision[indice_frame] = (
                frame.copy(),
                mascara,
                area,
                candidato
            )

        indice_frame += 1

    video_3.release()

    if not registros:
        raise ValueError("No se procesaron frames")


    # Tabla de mediciones

    tabla_mediciones_3 = pd.DataFrame(registros)

    tiempos_3 = tabla_mediciones_3["tiempo_s"].to_numpy()
    areas_3 = tabla_mediciones_3["area_px2"].to_numpy()
    diametros_3 = tabla_mediciones_3["diametro_px"].to_numpy()

    porcentaje_deteccion = (
        100 * tabla_mediciones_3["detectada"].mean()
    )

    print("\n--- 3.2.1 Procesamiento del video ---")
    print(f"Video: {Path(VIDEO_PATH).name}")
    print(f"Metodo de segmentacion: {METODO_VIDEO} + {OPERACION_VIDEO}")
    print(f"FPS: {fps_3:.2f}")
    print(f"Frames procesados: {len(registros)}")
    print(f"Duracion: {len(registros) / fps_3:.2f} s")
    print(f"Detecciones validas: {porcentaje_deteccion:.2f}%")

    print("\nMotivos por los que se descartaron intentos de deteccion:")
    if MOTIVOS_RECHAZO:
        for motivo, cantidad in MOTIVOS_RECHAZO.most_common():
            print(f"  {motivo}: {cantidad}")
    else:
        print("  (ninguno)")

    ruta_mediciones = (
        carpeta_resultados / f"mediciones_{nombre_video}.csv"
    )

    tabla_mediciones_3.to_csv(
        ruta_mediciones,
        index=False
    )

    print(f"Mediciones guardadas en: {ruta_mediciones}")

    # 3.2.2 Curvas temporales de area y diametro

    # Filtro de atipicos (Hampel): marca como NaN los valores que se alejan
    # demasiado de la mediana de su vecindad (picos aislados, segmentaciones
    # erroneas que pasaron los controles).

    def filtro_hampel(valores, ventana=7, n_sigmas=3.0, piso_rel=0.05):

        serie = pd.Series(valores, dtype=float)

        mediana = serie.rolling(
            ventana, center=True, min_periods=3
        ).median()

        desvio_abs = (serie - mediana).abs()

        mad = desvio_abs.rolling(
            ventana, center=True, min_periods=3
        ).median()

        # Umbral robusto, con un piso relativo para tramos casi constantes
        umbral = np.maximum(
            n_sigmas * 1.4826 * mad,
            piso_rel * mediana
        )

        atipicos = (desvio_abs > umbral).to_numpy()

        salida = serie.copy()
        salida[atipicos] = np.nan

        return salida.to_numpy(), atipicos


    # Interpolar solamente huecos de hasta 3 frames.

    def interpolar_huecos_cortos(valores, max_hueco=3):

        serie = pd.Series(valores, dtype=float)

        salida = serie.interpolate(
            method="linear",
            limit_area="inside"
        )

        es_nan = serie.isna()

        grupos = es_nan.ne(es_nan.shift()).cumsum()

        for _, bloque in serie.groupby(grupos):

            if bloque.isna().all() and len(bloque) > max_hueco:
                salida.loc[bloque.index] = np.nan

        return salida.to_numpy().copy()


    areas_sin_atipicos, mascara_atipicos = filtro_hampel(areas_3)

    print(f"Valores atipicos descartados (Hampel): {int(mascara_atipicos.sum())}")

    areas_interpoladas = interpolar_huecos_cortos(
        areas_sin_atipicos,
        max_hueco=3
    )

    # Mediana temporal de cinco frames
    serie_mediana = pd.Series(areas_interpoladas).rolling(
        window=5,
        center=True,
        min_periods=3
    ).median()

    areas_procesadas = serie_mediana.to_numpy().copy()

    # Mantener los intervalos sin mediciones
    areas_procesadas[~np.isfinite(areas_interpoladas)] = np.nan

    diametros_procesados = 2 * np.sqrt(
        areas_procesadas / np.pi
    )


    fig, ax = plt.subplots(
        2, 1,
        figsize=(11, 6),
        sharex=True
    )

    ax[0].plot(
        tiempos_3,
        areas_3,
        color="gray",
        alpha=0.5,
        label="Medicion original"
    )

    if mascara_atipicos.any():
        ax[0].scatter(
            tiempos_3[mascara_atipicos],
            areas_3[mascara_atipicos],
            color="red",
            s=14,
            zorder=3,
            label="Atipicos (Hampel)"
        )

    ax[0].plot(
        tiempos_3,
        areas_procesadas,
        color="blue",
        label="Serie procesada"
    )

    ax[0].set_title(
        f"Area pupilar en funcion del tiempo ({METODO_VIDEO} + {OPERACION_VIDEO})"
    )
    ax[0].set_ylabel("Area (px²)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(
        tiempos_3,
        diametros_3,
        color="gray",
        alpha=0.5
    )

    ax[1].plot(
        tiempos_3,
        diametros_procesados,
        color="blue"
    )

    ax[1].set_title("Diametro pupilar en funcion del tiempo")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Diametro equivalente (px)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



    # 3.2.3 PARAMETROS FISIOLOGICOS

    # Detectar automaticamente cuando se enciende la luz
    # y calcular los parametros a partir del area pupilar.

    # A. Medir la luminosidad a lo largo del video

    video_luz = cv2.VideoCapture(str(VIDEO_PATH))

    if not video_luz.isOpened():
        raise ValueError("No se pudo abrir el video para analizar la luz")

    luminosidades = []

    # Usamos una region fija alrededor de la semilla,
    # excluyendo la pupila para que sus cambios de tamaño
    # no se confundan con cambios de iluminacion.

    cx_semilla = SEMILLA["cx"]
    cy_semilla = SEMILLA["cy"]
    radio_semilla = SEMILLA["radio"]

    while True:
        ret, frame_luz = video_luz.read()

        if not ret:
            break

        gris_luz = cv2.cvtColor(
            frame_luz, cv2.COLOR_BGR2GRAY
        )

        alto_luz, ancho_luz = gris_luz.shape

        # Recorte local alrededor del ojo
        margen = int(np.ceil(3.2 * radio_semilla))

        x1 = max(0, int(cx_semilla - margen))
        x2 = min(ancho_luz, int(cx_semilla + margen + 1))
        y1 = max(0, int(cy_semilla - margen))
        y2 = min(alto_luz, int(cy_semilla + margen + 1))

        roi_luz = gris_luz[y1:y2, x1:x2]

        Y, X = np.ogrid[y1:y2, x1:x2]

        distancia = np.sqrt(
            (X - cx_semilla)**2 +
            (Y - cy_semilla)**2
        )

        # Anillo exterior a la pupila
        # para analizar principalmente iris y entorno.
        mascara_luz = (
            (distancia >= 2.0 * radio_semilla) &
            (distancia <= 3.0 * radio_semilla)
        )

        if np.any(mascara_luz):
            # La mediana reduce la influencia de reflejos
            # y pixeles excesivamente brillantes.
            brillo = np.median(roi_luz[mascara_luz])
        else:
            brillo = np.nan

        luminosidades.append(brillo)

    video_luz.release()

    luminosidades = np.asarray(luminosidades, dtype=float)
    tiempos_luz = np.arange(len(luminosidades)) / fps_3

    # Suavizado de la luminosidad
    ventana_luz = max(3, int(round(0.15 * fps_3)))
    if ventana_luz % 2 == 0:
        ventana_luz += 1

    luminosidad_suavizada = (
        pd.Series(luminosidades)
        .interpolate(limit_direction="both")
        .rolling(ventana_luz, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # B. Detectar el encendido de la luz

    # Comparamos la luminosidad media antes y despues
    # de cada instante, buscando un aumento sostenido.

    n_ventana = max(3, int(round(0.30 * fps_3)))

    cambios_luz = np.full(
        len(luminosidad_suavizada),
        np.nan
    )

    for i in range(n_ventana, len(cambios_luz) - n_ventana):

        brillo_antes = np.median(
            luminosidad_suavizada[i - n_ventana:i]
        )

        brillo_despues = np.median(
            luminosidad_suavizada[i:i + n_ventana]
        )

        cambios_luz[i] = brillo_despues - brillo_antes

    # Estimar nivel de ruido de los cambios de luminosidad
    cambios_validos = cambios_luz[np.isfinite(cambios_luz)]

    TIEMPO_ESTIMULO = np.nan
    TIEMPO_APAGADO = np.nan

    if len(cambios_validos) > 0:

        mediana_cambio = np.median(cambios_validos)
        mad_cambio = np.median(
            np.abs(cambios_validos - mediana_cambio)
        )

        ruido_luz = 1.4826 * mad_cambio

        # Exigimos un aumento superior al ruido y
        # al 8% del brillo típico de la imagen.
        umbral_cambio = max(
            4 * ruido_luz,
            0.08 * np.median(luminosidad_suavizada),
            5.0
        )

        # Buscar incrementos claros y sostenidos
        posibles_encendidos = np.where(
            cambios_luz > umbral_cambio
        )[0]

        # Ignoramos eventos demasiado cercanos al inicio
        # o al final, porque no tienen ventanas completas.
        posibles_encendidos = posibles_encendidos[
            (posibles_encendidos >= n_ventana) &
            (posibles_encendidos < len(cambios_luz) - n_ventana)
        ]

        if len(posibles_encendidos) > 0:

            # Agrupar frames consecutivos del mismo cambio.
            grupos = np.split(
                posibles_encendidos,
                np.where(np.diff(posibles_encendidos) > 1)[0] + 1
            )

            # Elegir el primer aumento significativo.
            primer_grupo = grupos[0]

            indice_estimulo = primer_grupo[
                np.argmax(cambios_luz[primer_grupo])
            ]

            TIEMPO_ESTIMULO = float(
                tiempos_luz[indice_estimulo]
            )

       
            # Detectar el apagado cuando la luminosidad
            # vuelve aproximadamente al nivel anterior al estimulo.

            # Nivel basal: mediana del segundo previo al encendido.
            basal_luz = (
                (tiempos_luz >= TIEMPO_ESTIMULO - 1.0) &
                (tiempos_luz < TIEMPO_ESTIMULO)
            )

            if np.any(basal_luz):

                brillo_basal = np.median(
                    luminosidad_suavizada[basal_luz]
                )

                # Nivel de iluminacion durante el estimulo,
                # evitando el pico transitorio del encendido.
                zona_encendida = (
                    (tiempos_luz >= TIEMPO_ESTIMULO + 0.7) &
                    (tiempos_luz <= TIEMPO_ESTIMULO + 2.0)
                )

                if np.any(zona_encendida):

                    brillo_encendido = np.median(
                        luminosidad_suavizada[zona_encendida]
                    )

                    # Umbral intermedio entre luz apagada y encendida.
                    umbral_apagado = (
                        brillo_basal +
                        0.35 * (brillo_encendido - brillo_basal)
                    )

                    if brillo_encendido > brillo_basal + umbral_cambio:

                        # Buscar un regreso sostenido al nivel oscuro.
                        # Se empieza despues de la estabilizacion inicial.
                        inicio_busqueda = TIEMPO_ESTIMULO + 2.0

                        n_confirmacion = max(
                            3, int(round(0.20 * fps_3))
                        )

                        for i in range(len(tiempos_luz) - n_confirmacion):

                            if tiempos_luz[i] < inicio_busqueda:
                                continue

                            tramo = luminosidad_suavizada[
                                i:i + n_confirmacion
                            ]

                            if np.all(tramo < umbral_apagado):

                                TIEMPO_APAGADO = float(
                                    tiempos_luz[i]
                                )
                                break


    print("\n--- Deteccion automatica del estimulo ---")

    if np.isfinite(TIEMPO_ESTIMULO):
        print(f"Encendido estimado: {TIEMPO_ESTIMULO:.3f} s")
    else:
        print("No se detecto un encendido luminoso confiable.")

    if np.isfinite(TIEMPO_APAGADO):
        print(f"Apagado estimado: {TIEMPO_APAGADO:.3f} s")

    # Visualizar la deteccion del estimulo
    plt.figure(figsize=(11, 4))

    plt.plot(
        tiempos_luz,
        luminosidad_suavizada,
        color="orange",
        label="Luminosidad"
    )

    if np.isfinite(TIEMPO_ESTIMULO):
        plt.axvline(
            TIEMPO_ESTIMULO,
            color="red",
            linestyle="--",
            label="Encendido estimado"
        )

    if np.isfinite(TIEMPO_APAGADO):
        plt.axvline(
            TIEMPO_APAGADO,
            color="blue",
            linestyle="--",
            label="Apagado estimado"
        )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Intensidad mediana (0-255)")
    plt.title("Deteccion automatica del estimulo luminoso")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # C. Calcular parametros fisiologicos

    latencia = np.nan
    amplitud = np.nan
    velocidad_constriccion = np.nan
    velocidad_redilatacion = np.nan

    area_basal = np.nan
    area_minima = np.nan
    tiempo_inicio = np.nan
    tiempo_minimo = np.nan

    estado_parametros = "No calculados"

    print("\n--- 3.2.3 Parametros fisiologicos ---")

    if not np.isfinite(TIEMPO_ESTIMULO):

        print("No se calculan parametros: estimulo no identificado.")
        estado_parametros = "Estimulo no detectado"

    else:

        t = np.asarray(tiempos_3, dtype=float)
        a = np.asarray(areas_procesadas, dtype=float)

        # Verificar calidad de las mediciones alrededor
        # del estimulo. No reemplazar automaticamente
        # intervalos grandes sin deteccion.
        basal = (
            (t >= TIEMPO_ESTIMULO - 1.0) &
            (t < TIEMPO_ESTIMULO) &
            np.isfinite(a)
        )

        # Limitar la búsqueda de constriccion a 4 segundos
        # o hasta el apagado de la luz, lo que ocurra primero.
        limite_constriccion = TIEMPO_ESTIMULO + 4.0

        if np.isfinite(TIEMPO_APAGADO):
            limite_constriccion = min(
                limite_constriccion,
                TIEMPO_APAGADO
            )

        respuesta = (
            (t >= TIEMPO_ESTIMULO) &
            (t <= limite_constriccion)
        )

        # Exigir mediciones suficientes y continuas
        # en el tramo de respuesta.
        proporcion_valida = (
            np.mean(np.isfinite(a[respuesta]))
            if np.any(respuesta) else 0
        )

        if basal.sum() < 3:

            estado_parametros = "Basal insuficiente"
            print("No hay suficientes mediciones antes del estimulo.")

        elif proporcion_valida < 0.90:

            estado_parametros = "Respuesta incompleta"
            print("Hay demasiadas mediciones faltantes durante la respuesta.")

        else:

            area_basal = float(np.median(a[basal]))

            indices_respuesta = np.where(
                respuesta & np.isfinite(a)
            )[0]

            tr = t[indices_respuesta]
            ar = a[indices_respuesta]

            if len(tr) < 5:

                estado_parametros = "Respuesta insuficiente"
                print("No hay suficientes mediciones posteriores.")

            else:

                # Area minima y amplitud
                i_min = int(np.argmin(ar))

                area_minima = float(ar[i_min])
                tiempo_minimo = float(tr[i_min])

                amplitud = area_basal - area_minima

                if amplitud <= 0:

                    estado_parametros = "Sin constriccion"
                    amplitud = np.nan
                    print("No se identifico una disminucion del area.")

                else:

                    # Inicio: descenso sostenido de al menos
                    # el 5% de la amplitud total.
                    umbral_inicio = (
                        area_basal - 0.05 * amplitud
                    )

                    debajo = ar <= umbral_inicio

                    indice_inicio = None

                    for i in range(max(0, i_min - 1)):

                        if (
                            i + 2 < len(ar) and
                            debajo[i] and
                            debajo[i + 1] and
                            debajo[i + 2]
                        ):
                            indice_inicio = i
                            break

                    if indice_inicio is None:

                        estado_parametros = "Inicio no identificado"
                        print("No se pudo identificar el inicio de constriccion.")

                    else:

                        tiempo_inicio = float(
                            tr[indice_inicio]
                        )

                        latencia = (
                            tiempo_inicio - TIEMPO_ESTIMULO
                        )

                        duracion_constriccion = (
                            tiempo_minimo - tiempo_inicio
                        )

                        if duracion_constriccion > 0:

                            velocidad_constriccion = (
                                amplitud / duracion_constriccion
                            )

                            # Redilatacion desde el minimo
                            # hasta la ultima muestra valida.
                            despues_minimo = (
                                (t > tiempo_minimo) &
                                np.isfinite(a)
                            )

                            if np.any(despues_minimo):

                                indices_despues = np.where(
                                    despues_minimo
                                )[0]

                                i_final = indices_despues[-1]

                                duracion_redilatacion = (
                                    t[i_final] - tiempo_minimo
                                )

                                if duracion_redilatacion > 0:
                                    velocidad_redilatacion = (
                                        (a[i_final] - area_minima) /
                                        duracion_redilatacion
                                    )

                            estado_parametros = "Calculados"

                            print(f"Area basal: {area_basal:.2f} px2")
                            print(f"Area minima: {area_minima:.2f} px2")
                            print(f"Amplitud: {amplitud:.2f} px2")
                            print(f"Latencia: {latencia:.3f} s")
                            print(
                                f"Velocidad de constriccion: "
                                f"{velocidad_constriccion:.2f} px2/s"
                            )
                            print(
                                f"Velocidad de redilatacion: "
                                f"{velocidad_redilatacion:.2f} px2/s"
                            )

                        else:

                            estado_parametros = "Duracion invalida"
                            print("Duracion de constriccion invalida.")

    print(f"Estado: {estado_parametros}")


    # D. Graficar los parametros sobre la curva pupilar

    plt.figure(figsize=(11, 4))

    plt.plot(
        tiempos_3,
        areas_procesadas,
        color="blue",
        label="Area pupilar procesada"
    )

    if np.isfinite(TIEMPO_ESTIMULO):
        plt.axvline(
            TIEMPO_ESTIMULO,
            color="red",
            linestyle="--",
            label="Estimulo"
        )

    if np.isfinite(tiempo_inicio):
        plt.axvline(
            tiempo_inicio,
            color="green",
            linestyle="--",
            label="Inicio de constriccion"
        )

    if np.isfinite(tiempo_minimo):
        plt.scatter(
            tiempo_minimo,
            area_minima,
            color="black",
            s=50,
            label="Area minima"
        )

    if np.isfinite(area_basal):
        plt.axhline(
            area_basal,
            color="gray",
            linestyle=":",
            label="Area basal"
        )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Area pupilar (px2)")
    plt.title("Respuesta pupilar al estimulo luminoso")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # E. Guardar parametros fisiologicos en CSV

    tabla_parametros = pd.DataFrame([{
        "video": Path(VIDEO_PATH).name,
        "metodo": METODO_VIDEO,
        "morfologia": OPERACION_VIDEO,
        "tiempo_estimulo_s": TIEMPO_ESTIMULO,
        "tiempo_apagado_s": TIEMPO_APAGADO,
        "area_basal_px2": area_basal,
        "area_minima_px2": area_minima,
        "tiempo_inicio_s": tiempo_inicio,
        "tiempo_minimo_s": tiempo_minimo,
        "latencia_s": latencia,
        "amplitud_px2": amplitud,
        "velocidad_constriccion_px2_s": velocidad_constriccion,
        "velocidad_redilatacion_px2_s": velocidad_redilatacion,
        "estado": estado_parametros
    }])

    ruta_parametros = (
        carpeta_resultados /
        f"parametros_fisiologicos_{nombre_video}.csv"
    )

    tabla_parametros.to_csv(
        ruta_parametros,
        index=False
    )

    print(f"\nParametros guardados en: {ruta_parametros}")



    # 3.2.4 ARTEFACTOS Y CONTROL DE CALIDAD

    # Visualizar los frames seleccionados y sus contornos


    # Mostrar frames en tandas para evitar superposiciones
    indices_mostrar = sorted(frames_revision)

    FRAMES_POR_TANDA = 10

    for inicio in range(0, len(indices_mostrar), FRAMES_POR_TANDA):

        tanda = indices_mostrar[inicio:inicio + FRAMES_POR_TANDA]

        fig, ax = plt.subplots(
            2,
            len(tanda),
            figsize=(15, 3.5),
            squeeze=False
        )

        for j, indice in enumerate(tanda):

            frame, mascara, area, candidato = frames_revision[indice]

            original_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            imagen_contorno = original_rgb.copy()

            contornos, _ = cv2.findContours(
                mascara,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                imagen_contorno,
                contornos,
                -1,
                (0, 255, 0),
                2
            )

            if candidato is not None:
                cv2.circle(
                    imagen_contorno,
                    (
                        int(round(candidato["cx"])),
                        int(round(candidato["cy"]))
                    ),
                    int(round(candidato["radio"])),
                    (255, 0, 0),
                    1
                )

            ax[0, j].imshow(original_rgb)
            ax[0, j].set_title(
                f"{indice / fps_3:.2f} s",
                fontsize=8
            )

            ax[1, j].imshow(imagen_contorno)

            if np.isfinite(area):
                titulo = f"{area:.0f} px²"
            else:
                titulo = "No detectada"

            ax[1, j].set_title(titulo, fontsize=8)

            ax[0, j].axis("off")
            ax[1, j].axis("off")

        plt.subplots_adjust(
            left=0.01,
            right=0.99,
            top=0.88,
            bottom=0.05,
            wspace=0.08,
            hspace=0.25
        )

        plt.show()
        plt.close(fig)



    frames_sin_deteccion = int(
        (tabla_mediciones_3["detectada"] == 0).sum()
    )

    print("\n--- 3.2.4 Control de calidad ---")
    print(f"Frames sin deteccion: {frames_sin_deteccion}")
    print(f"Porcentaje sin deteccion: {100 - porcentaje_deteccion:.2f}%")
    print(f"Atipicos descartados (Hampel): {int(mascara_atipicos.sum())}")
    print("Valores faltantes: NaN")
    print("Interpolacion: solo huecos de hasta 3 frames")
    print("Suavizado: mediana temporal de 5 frames")


    #
    # GUARDAR SERIE TEMPORAL
    #

    tabla_serie = pd.DataFrame({
        "tiempo_s": tiempos_3,
        "area_original_px2": areas_3,
        "area_procesada_px2": areas_procesadas,
        "diametro_original_px": diametros_3,
        "diametro_procesado_px": diametros_procesados
    })

    ruta_serie = (
        carpeta_resultados /
        f"serie_procesada_{nombre_video}.csv"
    )

    tabla_serie.to_csv(
        ruta_serie,
        index=False
    )

    print(f"Serie temporal guardada en: {ruta_serie}")

