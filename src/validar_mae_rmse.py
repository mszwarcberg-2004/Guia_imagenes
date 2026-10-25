"""Validacion manual (MAE / RMSE) del area y el diametro pupilar. Modulo 3.2.1.

Uso (desde la raiz del proyecto, despues de correr src/main.py):
    python src/validar_mae_rmse.py

Muestra 6 frames del video, uno por uno. En cada uno hay que hacer clic alrededor
del borde de la pupila y apretar ENTER. Si el ojo esta cerrado, ENTER sin marcar
para saltear el frame. Al final imprime MAE y RMSE y guarda una tabla en resultados/.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Rutas del proyecto
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"

VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "video5.mov"

N_FRAMES_VALIDACION = 6


def buscar_mediciones(video_path):
    """Busca las mediciones frame a frame que dejo main.py para este video.

    Primero en resultados/detalle/ (la mas reciente) y, si no hay, en el
    formato anterior: resultados/mediciones_<video>.csv
    """

    en_detalle = sorted(
        (CARPETA_RESULTADOS / "detalle").glob(
            f"mediciones_{video_path.stem}__*.csv"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if en_detalle:
        return en_detalle[0]

    return CARPETA_RESULTADOS / f"mediciones_{video_path.stem}.csv"


CSV_PATH = buscar_mediciones(VIDEO_PATH)


def calcular_errores(manual, automatico):
    manual = np.asarray(manual, dtype=float)
    automatico = np.asarray(automatico, dtype=float)

    diferencias = automatico - manual

    mae = np.mean(np.abs(diferencias))
    rmse = np.sqrt(np.mean(diferencias**2))

    return mae, rmse


def marcar_pupila(frame, indice, tiempo, cx, cy, diametro):
    # Recortar una region alrededor de la pupila para verla ampliada
    alto, ancho = frame.shape[:2]
    margen = max(50, int(1.8 * diametro))

    x1 = max(0, int(cx - margen))
    x2 = min(ancho, int(cx + margen))
    y1 = max(0, int(cy - margen))
    y2 = min(alto, int(cy + margen))

    recorte = frame[y1:y2, x1:x2]

    if recorte.size == 0:
        return None

    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(recorte_rgb)
    ax.set_title(
        f"Frame {indice} - Tiempo {tiempo:.2f} s\n"
        "Marca el borde pupilar con varios clics.\n"
        "Presiona ENTER para terminar."
    )

    print(f"\nFrame {indice} ({tiempo:.2f} s)")
    print("Marca puntos alrededor de la pupila y presiona Enter.")
    print("Si no se ve bien por un parpadeo, presiona Enter sin marcar.")

    plt.show(block=False)
    puntos = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(puntos) < 3:
        print("Frame omitido.")
        return None

    puntos = np.round(puntos).astype(np.int32)

    # Crear mascara manual
    mascara = np.zeros(recorte.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [puntos], 255)

    area_manual = int(np.count_nonzero(mascara))
    diametro_manual = 2 * np.sqrt(area_manual / np.pi)

    return area_manual, diametro_manual


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No existe el video: {VIDEO_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No existen las mediciones de {VIDEO_PATH.stem} en resultados/. "
            "Ejecuta primero src/main.py"
        )

    print(f"Mediciones automaticas: {CSV_PATH.name}")

    # Leer mediciones automaticas del Modulo 3
    tabla = pd.read_csv(CSV_PATH)

    # Elegir solamente frames con detecciones automaticas validas
    validos = tabla[
        (tabla["detectada"] == 1)
        & tabla["area_px2"].notna()
        & tabla["diametro_px"].notna()
        & tabla["centro_x"].notna()
        & tabla["centro_y"].notna()
    ].copy()

    if validos.empty:
        raise ValueError("No hay detecciones validas para comparar.")

    # Seleccionar frames distribuidos a lo largo del video
    cantidad = min(N_FRAMES_VALIDACION, len(validos))

    posiciones = np.linspace(
        0, len(validos) - 1, cantidad, dtype=int
    )

    seleccion = validos.iloc[posiciones]

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise ValueError("No se pudo abrir el video.")

    resultados = []

    try:
        for _, fila in seleccion.iterrows():
            indice = int(fila["frame"])
            tiempo = float(fila["tiempo_s"])

            video.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, frame = video.read()

            if not ok:
                print(f"No se pudo leer el frame {indice}")
                continue

            medicion = marcar_pupila(
                frame,
                indice,
                tiempo,
                float(fila["centro_x"]),
                float(fila["centro_y"]),
                float(fila["diametro_px"])
            )

            if medicion is None:
                continue

            area_manual, diametro_manual = medicion

            resultados.append({
                "frame": indice,
                "tiempo_s": tiempo,
                "area_manual_px2": area_manual,
                "area_automatica_px2": float(fila["area_px2"]),
                "diametro_manual_px": diametro_manual,
                "diametro_automatico_px": float(fila["diametro_px"])
            })

    finally:
        video.release()

    if not resultados:
        print("No se obtuvieron mediciones manuales.")
        return

    comparacion = pd.DataFrame(resultados)

    # Calcular errores de area
    mae_area, rmse_area = calcular_errores(
        comparacion["area_manual_px2"],
        comparacion["area_automatica_px2"]
    )

    # Calcular errores de diametro
    mae_diametro, rmse_diametro = calcular_errores(
        comparacion["diametro_manual_px"],
        comparacion["diametro_automatico_px"]
    )

    print("\n--- VALIDACION MANUAL VS AUTOMATICA ---")
    print(f"Frames comparados: {len(comparacion)}")
    print(f"MAE del area: {mae_area:.2f} px2")
    print(f"RMSE del area: {rmse_area:.2f} px2")
    print(f"MAE del diametro: {mae_diametro:.2f} px")
    print(f"RMSE del diametro: {rmse_diametro:.2f} px")

    # Guardar resultados
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    ruta_salida = (
        CARPETA_RESULTADOS / f"validacion_mae_rmse_{VIDEO_PATH.stem}.csv"
    )

    comparacion.round(4).to_csv(ruta_salida, index=False)

    print(f"\nComparacion guardada en: {ruta_salida}")

    # Grafico de comparacion de diametros
    plt.figure(figsize=(9, 5))

    plt.plot(
        comparacion["tiempo_s"],
        comparacion["diametro_manual_px"],
        "o-",
        label="Medicion manual"
    )

    plt.plot(
        comparacion["tiempo_s"],
        comparacion["diametro_automatico_px"],
        "s--",
        label="Medicion automatica"
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Diametro equivalente (px)")
    plt.title("Validacion del diametro pupilar")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()