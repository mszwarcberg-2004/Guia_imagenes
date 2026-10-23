
"""Validacion externa del TP PSIB: un video, sin seleccion manual.

Ubicar este archivo dentro de src/ del repositorio y editar solamente VIDEO_PATH.
Incluye una inicializacion automatica mas selectiva y reutiliza el modulo 3 del equipo.
"""

from pathlib import Path
import contextlib
import io
import traceback
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funciones_modulo3 import modulo3
from funciones_modulo4 import analizar_imagen, comparar_diametro_jpeg


RAIZ = Path(__file__).resolve().parent.parent

# ===== UNICA VARIABLE A MODIFICAR POR LA CATEDRA =====
VIDEO_PATH = RAIZ / "datos" / "Videos_ojo" / "video_prueba.mp4"
# Tambien se admite una ruta absoluta:
# VIDEO_PATH = Path("/Users/.../video.mov")
# ====================================================

METODO = "Watershed"
MORFOLOGIA = "Dilatacion"
SALIDAS = RAIZ / "resultados_externos"


# ============================================================
# ROI AUTOMATICA - MISMO DETECTOR QUE EL MODULO 3
# ============================================================

def _detectar_centro_pupila(gris, zona, rango_radios, n_radios=8):
    """
    Busca un disco oscuro y uniforme rodeado de un anillo
    mas claro, utilizando el criterio del Modulo 3.
    """

    ANCHO_DETECCION = 420
    PUNTAJE_MINIMO = 0.20
    PESO_UNIFORMIDAD = 0.5

    alto_original, ancho_original = gris.shape

    escala = min(
        1.0,
        ANCHO_DETECCION / ancho_original
    )

    ancho = max(
        1,
        int(round(ancho_original * escala))
    )

    alto = max(
        1,
        int(round(alto_original * escala))
    )

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

    rmin = max(
        3,
        int(round(rango_radios[0] * escala))
    )

    rmax = max(
        rmin,
        int(round(rango_radios[1] * escala))
    )

    radios = np.unique(
        np.round(
            np.linspace(rmin, rmax, n_radios)
        ).astype(int)
    )

    mejor = None
    mejor_puntaje = -np.inf

    for radio in radios:

        radio_externo = int(
            np.ceil(1.8 * radio)
        )

        coordenadas = np.arange(
            -radio_externo,
            radio_externo + 1
        )

        yy, xx = np.meshgrid(
            coordenadas,
            coordenadas,
            indexing="ij"
        )

        distancia = np.sqrt(
            xx ** 2 + yy ** 2
        )

        disco = (
            distancia <= radio
        ).astype(np.float32)

        anillo = (
            (distancia >= 1.2 * radio) &
            (distancia <= 1.8 * radio)
        ).astype(np.float32)

        disco /= disco.sum()
        anillo /= anillo.sum()

        media_disco = cv2.filter2D(
            imagen, -1, disco
        )

        media_disco_2 = cv2.filter2D(
            imagen_cuadrado, -1, disco
        )

        media_anillo = cv2.filter2D(
            imagen, -1, anillo
        )

        desvio_disco = np.sqrt(
            np.maximum(
                media_disco_2 - media_disco ** 2,
                0
            )
        )

        referencia = media_anillo + 10.0

        contraste_relativo = (
            media_anillo - media_disco
        ) / referencia

        puntaje_mapa = (
            contraste_relativo -
            PESO_UNIFORMIDAD *
            desvio_disco / referencia
        )

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

        puntaje = float(
            puntajes[indice]
        )

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


def preparar_roi_automatica(ruta, debug_dir=None):
    """
    Detecta la pupila en 15 frames del video.

    Usa el mismo criterio de consenso espacial del
    Modulo 3 y devuelve una ROI y mascara iniciales
    centradas en la pupila.
    """

    cap = cv2.VideoCapture(str(ruta))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"No se pudo abrir el video: {ruta}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if (
        not np.isfinite(fps)
        or fps <= 0
        or total <= 0
    ):
        cap.release()
        raise ValueError(
            "FPS o cantidad de frames invalidos"
        )

    indices = np.linspace(
        0,
        total - 1,
        15,
        dtype=int
    )

    candidatos = []

    try:

        for indice in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(indice)
            )

            ok, frame = cap.read()

            if not ok:
                continue

            gris = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            alto, ancho = gris.shape

            zona = (
                0.10 * ancho,
                0.90 * ancho,
                0.15 * alto,
                0.85 * alto
            )

            rango = (
                0.012 * ancho,
                0.08 * ancho
            )

            candidato = _detectar_centro_pupila(
                gris,
                zona,
                rango,
                n_radios=8
            )

            if candidato is not None:

                candidato["indice"] = int(indice)

                candidatos.append(candidato)

    finally:

        cap.release()

    if len(candidatos) < 3:
        raise ValueError(
            "No se encontraron suficientes candidatos "
            "para localizar la pupila automaticamente"
        )

    # Consenso espacial: mismo criterio del Modulo 3

    cxs = np.array([
        c["cx"] for c in candidatos
    ])

    cys = np.array([
        c["cy"] for c in candidatos
    ])

    radios = np.array([
        c["radio"] for c in candidatos
    ])

    mediana_x = np.median(cxs)
    mediana_y = np.median(cys)
    mediana_r = np.median(radios)

    cerca = (
        np.hypot(
            cxs - mediana_x,
            cys - mediana_y
        ) <= mediana_r
    )

    if cerca.sum() < 3:
        raise ValueError(
            "No se encontro una posicion pupilar "
            "consistente entre los frames"
        )

    validos = [
        c
        for c, es_cercano in zip(candidatos, cerca)
        if es_cercano
    ]

    # Elegir una deteccion representativa del consenso

    cx_med = float(
        np.median([c["cx"] for c in validos])
    )

    cy_med = float(
        np.median([c["cy"] for c in validos])
    )

    radio_med = float(
        np.median([c["radio"] for c in validos])
    )

    elegido = min(
        validos,
        key=lambda c: (
            np.hypot(
                c["cx"] - cx_med,
                c["cy"] - cy_med
            )
            + abs(c["radio"] - radio_med)
        )
    )

    cap = cv2.VideoCapture(str(ruta))

    try:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            elegido["indice"]
        )

        ok, frame = cap.read()

    finally:

        cap.release()

    if not ok:
        raise ValueError(
            "No se pudo recuperar el frame seleccionado"
        )

    alto, ancho = frame.shape[:2]

    cx = int(round(elegido["cx"]))
    cy = int(round(elegido["cy"]))

    radio = float(elegido["radio"])

    # Recorte con margen suficiente para ver
    # la pupila y parte del iris.

    margen = max(
        35,
        int(round(2.5 * radio))
    )

    x1 = max(0, cx - margen)
    x2 = min(ancho, cx + margen + 1)

    y1 = max(0, cy - margen)
    y2 = min(alto, cy + margen + 1)

    roi = frame[y1:y2, x1:x2].copy()

    if roi.size == 0:
        raise ValueError(
            "La ROI inicial quedo vacia"
        )

    roi_gris = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Reutilizar la segmentacion que ya probamos
    # con la ROI corregida y que te dio buen resultado.

    mascara = crear_mascara_corregida(roi)

    # Guardar diagnostico sobre el frame completo

    if debug_dir is not None:

        debug_dir = Path(debug_dir)

        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        vista = frame.copy()

        cv2.rectangle(
            vista,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2
        )

        cv2.circle(
            vista,
            (cx, cy),
            max(1, int(round(radio))),
            (0, 255, 255),
            2
        )

        cv2.imwrite(
            str(
                debug_dir /
                "seleccion_roi_sobre_frame.jpg"
            ),
            vista
        )

    print(
        f"Semilla inicial: centro=({cx}, {cy}), "
        f"radio={radio:.1f} px"
    )

    print(
        f"Frames consistentes: "
        f"{len(validos)}/{len(indices)}"
    )

    segundo = elegido["indice"] / fps

    return (
        frame,
        roi,
        roi_gris,
        mascara,
        segundo
    )



# ============================================================
# GRAFICOS
# ============================================================

def guardar_curva(serie, destino):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True
    )

    t = serie["tiempo_s"]

    axes[0].plot(
        t,
        serie["area_original_px2"],
        alpha=0.3,
        label="Area detectada"
    )

    axes[0].plot(
        t,
        serie["area_procesada_px2"],
        label="Area procesada"
    )

    axes[0].set_ylabel("Area (px²)")

    axes[1].plot(
        t,
        serie["diametro_original_px"],
        alpha=0.3,
        label="Diametro detectado"
    )

    axes[1].plot(
        t,
        serie["diametro_procesado_px"],
        label="Diametro procesado"
    )

    axes[1].set_ylabel(
        "Diametro equivalente (px)"
    )

    axes[1].set_xlabel("Tiempo (s)")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Seguimiento pupilar automatico"
    )

    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    plt.close(fig)


# ============================================================
# VIDEO CON DETECCION SUPERPUESTA
# ============================================================

def guardar_video_revision(video_path, mediciones, destino):

    """
    Dibuja el centro y el circulo equivalente.

    IMPORTANTE: No es el contorno segmentado exacto.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir el video para la vista previa"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    ancho = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    alto = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if (
        not np.isfinite(fps)
        or fps <= 0
        or ancho <= 0
        or alto <= 0
    ):
        cap.release()
        raise RuntimeError(
            "FPS o dimensiones invalidas"
        )

    writer = cv2.VideoWriter(
        str(destino),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ancho, alto)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "No fue posible crear el video MP4"
        )

    datos = mediciones.set_index(
        "frame",
        drop=False
    )

    idx = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            if idx in datos.index:

                m = datos.loc[idx]

                if isinstance(m, pd.DataFrame):
                    m = m.iloc[0]

                valido = (
                    int(m["detectada"]) == 1
                    and all(
                        pd.notna(m[k])
                        for k in (
                            "centro_x",
                            "centro_y",
                            "diametro_px"
                        )
                    )
                )

                if valido:

                    centro = (
                        int(round(m["centro_x"])),
                        int(round(m["centro_y"]))
                    )

                    radio = max(
                        1,
                        int(
                            round(
                                float(m["diametro_px"]) / 2
                            )
                        )
                    )

                    cv2.circle(
                        frame,
                        centro,
                        radio,
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        centro,
                        2,
                        (0, 255, 0),
                        -1
                    )

                    rotulo = (
                        f"Frame {idx}: "
                        f"area {m['area_px2']:.0f} px2"
                    )

                else:

                    rotulo = (
                        f"Frame {idx}: SIN DETECCION"
                    )

            else:

                rotulo = (
                    f"Frame {idx}: SIN MEDICION"
                )

            cv2.rectangle(
                frame,
                (0, 0),
                (min(ancho, 530), 33),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                rotulo,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            idx += 1

    finally:
        cap.release()
        writer.release()



# ============================================================
# ROI CORREGIDA A PARTIR DEL SEGUIMIENTO DEL MODULO 3
# ============================================================

def guardar_roi_desde_seguimiento(video_path, mediciones, carpeta):
    """
    Obtiene una ROI centrada en una deteccion valida del
    seguimiento pupilar del Modulo 3.

    No utiliza la ROI inicial que selecciono HoughCircles.
    """

    validas = mediciones[
        mediciones["detectada"] == 1
    ].copy()

    validas = validas.dropna(
        subset=["centro_x", "centro_y", "diametro_px"]
    )

    validas = validas[
        validas["diametro_px"] > 0
    ]

    if validas.empty:
        raise RuntimeError(
            "No hay detecciones validas para obtener la ROI corregida"
        )

    # Elegir un frame representativo dentro del seguimiento
    fila = validas.iloc[len(validas) // 2]

    indice = int(fila["frame"])
    cx = int(round(float(fila["centro_x"])))
    cy = int(round(float(fila["centro_y"])))
    diametro = float(fila["diametro_px"])

    radio = diametro / 2.0

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            "No se pudo leer el frame seleccionado"
        )

    alto, ancho = frame.shape[:2]

    if not (0 <= cx < ancho and 0 <= cy < alto):
        raise RuntimeError(
            "El centro detectado esta fuera del frame"
        )

    # Incluir la pupila y parte del iris
    margen = max(35, int(round(2.5 * radio)))

    x1 = max(0, cx - margen)
    x2 = min(ancho, cx + margen + 1)

    y1 = max(0, cy - margen)
    y2 = min(alto, cy + margen + 1)

    roi_corregida = frame[y1:y2, x1:x2].copy()

    if roi_corregida.size == 0:
        raise RuntimeError("La ROI corregida quedo vacia")

    # Guardar recorte corregido
    cv2.imwrite(
        str(carpeta / "roi_corregida.png"),
        roi_corregida
    )

    # Mostrar donde se ubica la ROI sobre el frame
    vista = frame.copy()

    cv2.rectangle(
        vista,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        2
    )

    cv2.circle(
        vista,
        (cx, cy),
        max(1, int(round(radio))),
        (0, 255, 0),
        2
    )

    cv2.circle(
        vista,
        (cx, cy),
        3,
        (0, 255, 0),
        -1
    )

    cv2.imwrite(
        str(carpeta / "roi_corregida_sobre_frame.png"),
        vista
    )

    print(
        f"ROI corregida guardada usando el frame {indice}"
    )

    return roi_corregida


# ============================================================
# MASCARA DE LA ROI CORREGIDA
# ============================================================

def crear_mascara_corregida(roi_corregida):
    """
    Segmenta la pupila dentro de la ROI obtenida
    del seguimiento del Modulo 3.

    Devuelve una mascara binaria:
    blanco = pupila
    negro = resto de la imagen
    """

    gris = cv2.cvtColor(
        roi_corregida,
        cv2.COLOR_BGR2GRAY
    )

    gris_suave = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    alto, ancho = gris.shape

    cx = ancho // 2
    cy = alto // 2

    # Limitamos la busqueda a la region central
    # para evitar confundir iris y parpados con pupila.
    radio_busqueda = max(
        10,
        int(min(alto, ancho) * 0.32)
    )

    yy, xx = np.ogrid[:alto, :ancho]

    zona_central = (
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ) <= radio_busqueda ** 2

    # La pupila suele ser la region mas oscura.
    radio_semilla = max(
        4,
        int(min(alto, ancho) * 0.09)
    )

    zona_semilla = (
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ) <= radio_semilla ** 2

    intensidad_semilla = float(
        np.median(gris_suave[zona_semilla])
    )

    # Umbral conservador: cerca de la intensidad
    # de la pupila, para no incluir todo el iris.
    umbral = min(
        100,
        intensidad_semilla + 25
    )

    binaria = (
        (gris_suave <= umbral) &
        zona_central
    ).astype(np.uint8)

    # Unir pequeños huecos
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    binaria = cv2.morphologyEx(
        binaria,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Conservar solamente la componente
    # oscura conectada al centro de la ROI.
    cantidad, etiquetas, stats, centroides = (
        cv2.connectedComponentsWithStats(
            binaria,
            8
        )
    )

    componente = int(etiquetas[cy, cx])

    if componente == 0:
        raise RuntimeError(
            "No se encontro una region oscura "
            "en el centro de la ROI corregida"
        )

    area = int(
        stats[componente, cv2.CC_STAT_AREA]
    )

    if area < 50:
        raise RuntimeError(
            "La mascara corregida es demasiado pequena"
        )

    mascara = np.where(
        etiquetas == componente,
        255,
        0
    ).astype(np.uint8)

    print(
        f"Mascara corregida generada. "
        f"Area: {area} px2"
    )

    return mascara

# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

def ejecutar():

    video = Path(VIDEO_PATH).expanduser().resolve()

    if not video.is_file():
        raise FileNotFoundError(
            f"No existe el video: {video}"
        )

    # Crear carpeta de resultados
    carpeta = SALIDAS / video.stem
    carpeta.mkdir(parents=True, exist_ok=True)

    resultados = carpeta / "resultados"
    resultados.mkdir(exist_ok=True)

    log_path = carpeta / "ejecucion.txt"

    print(f"Video seleccionado: {video}")

    # ========================================================
    # 1. ROI AUTOMATICA INICIAL
    # ========================================================

    print("\nInicializando ROI automatica...")

    frame, roi, roi_gris, mascara, segundo = (
        preparar_roi_automatica(
            video,
            debug_dir=carpeta / "diagnostico_roi"
        )
    )

    cv2.imwrite(
        str(carpeta / "roi_detectada.png"),
        roi
    )

    cv2.imwrite(
        str(carpeta / "mascara_inicial.png"),
        mascara
    )

    print(f"ROI inicial: {segundo:.2f} s")

    # ========================================================
    # 2. MODULO 3
    # ========================================================

    print("\nProcesando Modulo 3...")

    log = io.StringIO()

    mostrar_original = plt.show
    plt.show = lambda *args, **kwargs: plt.close("all")

    try:

        with contextlib.redirect_stdout(log):
            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    RuntimeWarning
                )

                modulo3(
                    frame,
                    roi,
                    roi_gris,
                    mascara,
                    METODO,
                    MORFOLOGIA,
                    video,
                    carpeta
                )

    finally:

        plt.show = mostrar_original
        plt.close("all")

        log_path.write_text(
            log.getvalue(),
            encoding="utf-8"
        )

    # ========================================================
    # 3. LEER LOS RESULTADOS DEL MODULO 3
    # ========================================================

    serie_csv = (
        resultados /
        f"serie_procesada_{video.stem}.csv"
    )

    mediciones_csv = (
        resultados /
        f"mediciones_{video.stem}.csv"
    )

    parametros_csv = (
        resultados /
        f"parametros_fisiologicos_{video.stem}.csv"
    )

    for archivo in [
        serie_csv,
        mediciones_csv,
        parametros_csv
    ]:
        if not archivo.exists():
            raise RuntimeError(
                f"No se genero {archivo.name}"
            )

    serie = pd.read_csv(serie_csv)
    mediciones = pd.read_csv(mediciones_csv)
    parametros = pd.read_csv(parametros_csv)

    if serie.empty or mediciones.empty or parametros.empty:
        raise RuntimeError("Se genero un CSV vacio")

    # ========================================================
    # 4. GUARDAR CURVAS Y PARAMETROS
    # ========================================================

    temporal = serie.merge(
        mediciones[
            [
                "frame",
                "tiempo_s",
                "detectada",
                "centro_x",
                "centro_y",
                "puntaje_deteccion"
            ]
        ],
        on="tiempo_s",
        how="left",
        validate="one_to_one"
    )

    temporal = temporal.rename(
        columns={
            "area_procesada_px2": "area_px2",
            "diametro_procesado_px": "diametro_px"
        }
    )

    temporal.to_csv(
        carpeta / "curva_pupilar.csv",
        index=False
    )

    parametros.to_csv(
        carpeta / "parametros_fisiologicos.csv",
        index=False
    )

    guardar_curva(
        serie,
        carpeta / "curva_pupilar.png"
    )

    # ========================================================
    # 5. OBTENER ROI CORREGIDA
    # ========================================================

    print("\nObteniendo ROI desde el seguimiento...")

   
    try:

        roi_corregida = guardar_roi_desde_seguimiento(
            video,
            mediciones,
            carpeta
        )

        # Crear mascara de la pupila real
        mascara_corregida = crear_mascara_corregida(
            roi_corregida
        )

        # Guardar mascara corregida
        cv2.imwrite(
            str(carpeta / "mascara_corregida.png"),
            mascara_corregida
        )

        print(
            "ROI y mascara corregidas guardadas correctamente."
        )

    except Exception as exc:

        print(
            f"Aviso: no se pudo corregir ROI o mascara: {exc}"
        )

        roi_corregida = None
        mascara_corregida = None

    # ========================================================
    # 6. VIDEO DE REVISION
    # ========================================================

    print("\nGenerando video de revision...")

    try:

        guardar_video_revision(
            video,
            mediciones,
            carpeta / "revision_pupila.mp4"
        )

    except Exception as exc:

        print(
            f"No se pudo generar el video de revision: {exc}"
        )

    # ========================================================
    # 7. MODULO 4
    # ========================================================

    print("\nProcesando Modulo 4...")

    log4 = io.StringIO()

    plt.show = lambda *args, **kwargs: plt.close("all")

    try:

        with contextlib.redirect_stdout(log4):

            if roi_corregida is not None:

                roi_corregida_gris = cv2.cvtColor(
                    roi_corregida,
                    cv2.COLOR_BGR2GRAY
                )

                analizar_imagen(
                    roi_corregida_gris,
                    nombre="ROI corregida externa"
                )

                comparar_diametro_jpeg(
                    roi_corregida_gris
                )

            else:

                print(
                    "ROI corregida no disponible. "
                    "Se omite su analisis."
                )

            # La mascara inicial proviene de otra
            # deteccion, por lo que no la usamos
            # como mascara de la ROI corregida.
           
            if mascara_corregida is not None:

                analizar_imagen(
                    mascara_corregida,
                    nombre="Mascara pupilar corregida"
                )

                print(
                    "Mascara corregida incluida "
                    "en el analisis del Modulo 4."
                )


            # Frame completo del video externo
            frame_gris = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            analizar_imagen(
                frame_gris,
                nombre="Frame externo"
            )

    finally:

        plt.show = mostrar_original
        plt.close("all")

        (
            carpeta / "modulo4_compresion.txt"
        ).write_text(
            log4.getvalue(),
            encoding="utf-8"
        )

    # ========================================================
    # 8. RESUMEN FINAL
    # ========================================================

    estado = str(
        parametros.iloc[0].get(
            "estado",
            "Desconocido"
        )
    )

    print("\n===== RESULTADOS =====")

    print(f"Estado de parametros: {estado}")

    print(
        f"Detecciones: "
        f"{int(mediciones['detectada'].sum())}"
        f"/{len(mediciones)} frames"
    )

    for campo in [
        "latencia_s",
        "amplitud_px2",
        "velocidad_constriccion_px2_s",
        "velocidad_redilatacion_px2_s"
    ]:

        print(
            f"{campo}: "
            f"{parametros.iloc[0].get(campo, np.nan)}"
        )

    print(f"\nResultados guardados en: {carpeta}")

    print(
        "IMPORTANTE: revisar visualmente la ROI "
        "y el seguimiento antes de interpretar "
        "los parametros fisiologicos."
    )


# ============================================================
# INICIAR EL PROGRAMA
# ============================================================

if __name__ == "__main__":

    try:
        ejecutar()

    except Exception as exc:

        print(
            f"ERROR EN VALIDACION EXTERNA: {exc}"
        )

        traceback.print_exc()

        raise SystemExit(1)
