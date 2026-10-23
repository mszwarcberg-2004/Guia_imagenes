
# MODULO 4 - COMPRESION DE IMAGENES
# Huffman, RLE y JPEG

import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from skimage.metrics import structural_similarity as ssim


# -------------------------------------------
# 4.1 COMPRESION HUFFMAN
# -------------------------------------------

def calcular_huffman(imagen):
    """
    Calcula el tamaño teorico de la imagen
    codificada mediante Huffman.
    """

    pixeles = imagen.flatten().tolist()
    frecuencias = Counter(pixeles)

    heap = []
    contador = 0

    for valor, frecuencia in frecuencias.items():
        heapq.heappush(
            heap, (frecuencia, contador, int(valor))
        )
        contador += 1

    if len(frecuencias) == 1:
        bits_huffman = len(pixeles)

    else:
        while len(heap) > 1:

            freq1, _, nodo1 = heapq.heappop(heap)
            freq2, _, nodo2 = heapq.heappop(heap)

            nuevo_nodo = (nodo1, nodo2)

            heapq.heappush(
                heap,
                (freq1 + freq2, contador, nuevo_nodo)
            )

            contador += 1

        raiz = heap[0][2]
        longitudes = {}

        def recorrer_arbol(nodo, profundidad=0):

            if isinstance(nodo, int):
                longitudes[nodo] = max(1, profundidad)

            else:
                recorrer_arbol(nodo[0], profundidad + 1)
                recorrer_arbol(nodo[1], profundidad + 1)

        recorrer_arbol(raiz)

        bits_huffman = sum(
            frecuencias[valor] * longitudes[int(valor)]
            for valor in frecuencias
        )

    bits_original = imagen.size * 8
    ratio = bits_original / bits_huffman

    return bits_huffman, ratio


# -------------------------------------------
# 4.1 COMPRESION RLE
# -------------------------------------------

def calcular_rle(secuencia):
    """
    Codifica las repeticiones consecutivas
    de cada intensidad.
    """

    secuencia = np.asarray(secuencia).flatten()

    if len(secuencia) == 0:
        return []

    resultado = []

    valor_actual = int(secuencia[0])
    repeticiones = 1

    for valor in secuencia[1:]:

        valor = int(valor)

        if valor == valor_actual and repeticiones < 65535:
            repeticiones += 1

        else:
            resultado.append(
                (valor_actual, repeticiones)
            )

            valor_actual = valor
            repeticiones = 1

    resultado.append(
        (valor_actual, repeticiones)
    )

    return resultado


def comparar_rle(imagen):
    """
    Compara RLE horizontal y vertical.
    Cada par utiliza 8 bits para la intensidad
    y 16 bits para la cantidad de repeticiones.
    """

    secuencia_horizontal = imagen.flatten(order="C")
    secuencia_vertical = imagen.flatten(order="F")

    rle_horizontal = calcular_rle(secuencia_horizontal)
    rle_vertical = calcular_rle(secuencia_vertical)

    bits_original = imagen.size * 8

    bits_horizontal = len(rle_horizontal) * 24
    bits_vertical = len(rle_vertical) * 24

    ratio_horizontal = bits_original / bits_horizontal
    ratio_vertical = bits_original / bits_vertical

    return {
        "horizontal": {
            "pares": len(rle_horizontal),
            "bits": bits_horizontal,
            "ratio": ratio_horizontal
        },
        "vertical": {
            "pares": len(rle_vertical),
            "bits": bits_vertical,
            "ratio": ratio_vertical
        }
    }


# -------------------------------------------
# 4.1 COMPARACION HUFFMAN Y RLE
# -------------------------------------------

def comparar_compresion_sin_perdida(imagen, nombre="Imagen"):

    imagen = np.asarray(imagen, dtype=np.uint8)

    bits_huffman, ratio_huffman = calcular_huffman(imagen)

    resultados_rle = comparar_rle(imagen)

    print(f"\n--- COMPRESION SIN PERDIDA: {nombre} ---")

    print(f"Dimensiones: {imagen.shape}")
    print(f"Tamaño original: {imagen.size * 8} bits")

    print("\nHuffman:")
    print(f"Tamaño codificado: {bits_huffman} bits")
    print(f"Ratio teorico: {ratio_huffman:.3f}")

    print("\nRLE horizontal:")
    print(
        f"Cantidad de pares: "
        f"{resultados_rle['horizontal']['pares']}"
    )
    print(
        f"Ratio teorico: "
        f"{resultados_rle['horizontal']['ratio']:.3f}"
    )

    print("\nRLE vertical:")
    print(
        f"Cantidad de pares: "
        f"{resultados_rle['vertical']['pares']}"
    )
    print(
        f"Ratio teorico: "
        f"{resultados_rle['vertical']['ratio']:.3f}"
    )

    return {
        "Huffman": ratio_huffman,
        "RLE horizontal": resultados_rle["horizontal"]["ratio"],
        "RLE vertical": resultados_rle["vertical"]["ratio"]
    }


# -------------------------------------------
# 4.2 COMPRESION JPEG
# -------------------------------------------

def comparar_jpeg(imagen, nombre="Imagen"):
    """
    Comprime una imagen con Q=90, 50, 10 y 5.
    Calcula ratio, PSNR y SSIM.
    """

    imagen = np.asarray(imagen, dtype=np.uint8)

    calidades = [90, 50, 10, 5]
    resultados = []

    bytes_original = imagen.nbytes

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))

    ax[0].imshow(
        imagen,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    ax[0].set_title("Original")
    ax[0].axis("off")

    print(f"\n--- COMPRESION JPEG: {nombre} ---")

    for i, calidad in enumerate(calidades):

        # Comprimir a JPEG
        exito, jpeg = cv2.imencode(
            ".jpg",
            imagen,
            [cv2.IMWRITE_JPEG_QUALITY, calidad]
        )

        if not exito:
            raise RuntimeError("No se pudo comprimir la imagen")

        # Reconstruir la imagen
        reconstruida = cv2.imdecode(
            jpeg,
            cv2.IMREAD_GRAYSCALE
        )

        # Tamaño real del JPEG
        bytes_jpeg = len(jpeg)

        # Ratio de compresion
        ratio = bytes_original / bytes_jpeg

        # Metricas de calidad
        psnr_valor = cv2.PSNR(
            imagen,
            reconstruida
        )

        ssim_valor = ssim(
            imagen,
            reconstruida,
            data_range=255
        )

        resultados.append({
            "Q": calidad,
            "bytes": bytes_jpeg,
            "ratio": ratio,
            "PSNR": psnr_valor,
            "SSIM": ssim_valor
        })

        print(f"\nJPEG Q={calidad}")
        print(f"Tamaño: {bytes_jpeg} bytes")
        print(f"Ratio: {ratio:.3f}")
        print(f"PSNR: {psnr_valor:.2f} dB")
        print(f"SSIM: {ssim_valor:.4f}")

        # Mostrar imagen reconstruida
        ax[i + 1].imshow(
            reconstruida,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax[i + 1].set_title(f"JPEG Q={calidad}")
        ax[i + 1].axis("off")

    plt.suptitle(f"Comparacion JPEG - {nombre}")
    plt.tight_layout()
    plt.show()

    return resultados


# -------------------------------------------
# 4.3 GRAFICOS PSNR Y SSIM VS COMPRESION
# -------------------------------------------

def graficar_metricas_jpeg(resultados, nombre="Imagen"):
    """
    Grafica PSNR y SSIM en funcion
    del ratio de compresion JPEG.
    """

    ratios = [r["ratio"] for r in resultados]
    valores_psnr = [r["PSNR"] for r in resultados]
    valores_ssim = [r["SSIM"] for r in resultados]
    calidades = [r["Q"] for r in resultados]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # PSNR
    ax[0].plot(ratios, valores_psnr, "o-")
    ax[0].set_xlabel("Ratio de compresion")
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_title("PSNR vs. compresion")
    ax[0].grid(True)

    # SSIM
    ax[1].plot(ratios, valores_ssim, "o-")
    ax[1].set_xlabel("Ratio de compresion")
    ax[1].set_ylabel("SSIM")
    ax[1].set_title("SSIM vs. compresion")
    ax[1].grid(True)

    # Mostrar el factor Q de cada punto
    for i, calidad in enumerate(calidades):

        ax[0].annotate(
            f"Q={calidad}",
            (ratios[i], valores_psnr[i]),
            xytext=(4, 5),
            textcoords="offset points"
        )

        ax[1].annotate(
            f"Q={calidad}",
            (ratios[i], valores_ssim[i]),
            xytext=(4, 5),
            textcoords="offset points"
        )

    plt.suptitle(
        f"Calidad vs. compresion - {nombre}"
    )
    plt.tight_layout()
    plt.show()


# -------------------------------------------
# 4.4 ANALISIS COMPLETO DE UNA IMAGEN
# -------------------------------------------

def analizar_imagen(imagen, nombre):
    """
    Ejecuta Huffman, RLE y JPEG,
    y grafica las metricas de calidad.
    """

    if imagen is None:
        print(f"\nNo se encontro: {nombre}")
        return

    comparar_compresion_sin_perdida(
        imagen,
        nombre=nombre
    )

    resultados_jpeg = comparar_jpeg(
        imagen,
        nombre=nombre
    )

    graficar_metricas_jpeg(
        resultados_jpeg,
        nombre=nombre
    )

    return resultados_jpeg


# -------------------------------------------
# PRUEBA DEL MODULO 4
# -------------------------------------------

if __name__ == "__main__":

    CARPETA_PROYECTO = Path(__file__).resolve().parent.parent

    # ---------------------------------------
    # A. ROI DEL OJO
    # ---------------------------------------

    ruta_roi = (
        CARPETA_PROYECTO / "imagenes" / "roi.png"
    )

    roi = cv2.imread(
        str(ruta_roi),
        cv2.IMREAD_GRAYSCALE
    )

    if roi is None:
        raise FileNotFoundError(
            f"No se encontro la ROI: {ruta_roi}"
        )

    analizar_imagen(
        roi,
        nombre="ROI del ojo"
    )

    # ---------------------------------------
    # B. MASCARA PUPILAR
    # ---------------------------------------

    ruta_mascara = (
        CARPETA_PROYECTO
        / "resultados"
        / "mascara_pupila_final.png"
    )

    mascara = cv2.imread(
        str(ruta_mascara),
        cv2.IMREAD_GRAYSCALE
    )

    if mascara is not None:

        # Asegurar mascara binaria
        _, mascara = cv2.threshold(
            mascara,
            127,
            255,
            cv2.THRESH_BINARY
        )

        analizar_imagen(
            mascara,
            nombre="Mascara pupilar"
        )

    else:
        print("\nNo se encontro la mascara pupilar.")
        print("Se omite su analisis de compresion.")

    # ---------------------------------------
    # C. FRAME COMPLETO
    # ---------------------------------------

    ruta_frame = (
        CARPETA_PROYECTO / "imagenes" / "frame.png"
    )

    frame = cv2.imread(
        str(ruta_frame),
        cv2.IMREAD_GRAYSCALE
    )

    if frame is not None:

        analizar_imagen(
            frame,
            nombre="Frame completo"
        )

    else:
        print("\nNo se encontro el frame completo.")
        print("Se omite su analisis de compresion.")

    print("\n--- FIN DEL ANALISIS DEL MODULO 4 ---")
def estimar_diametro_jpeg(imagen):
    """
    Segmenta la pupila mediante Watershed + dilatacion
    y calcula su diametro equivalente.
    """
    from funciones_modulo2 import segmentar_watershed

    _, mascara_otsu = cv2.threshold(
        imagen, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    _, mascara_ws = segmentar_watershed(imagen, mascara_otsu)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mascara = cv2.dilate(mascara_ws, kernel, iterations=1)

    # Conservar la componente de mayor area
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return np.nan

    contorno = max(contornos, key=cv2.contourArea)
    mascara_limpia = np.zeros_like(mascara)
    cv2.drawContours(
        mascara_limpia, [contorno], -1, 255, cv2.FILLED
    )

    area = np.count_nonzero(mascara_limpia)
    diametro = 2 * np.sqrt(area / np.pi)

    return diametro


def comparar_diametro_jpeg(roi_gris):
    """Compara el diametro original con JPEG Q=50 y Q=10."""

    diametro_original = estimar_diametro_jpeg(roi_gris)

    print("\n--- EFECTO DE JPEG SOBRE EL DIAMETRO ---")
    print(f"Original: {diametro_original:.2f} px")

    for calidad in [50, 10]:

        exito, jpeg = cv2.imencode(
            ".jpg", roi_gris,
            [cv2.IMWRITE_JPEG_QUALITY, calidad]
        )

        if not exito:
            raise RuntimeError("No se pudo generar JPEG")

        reconstruida = cv2.imdecode(
            jpeg, cv2.IMREAD_GRAYSCALE
        )

        diametro = estimar_diametro_jpeg(reconstruida)

        diferencia = diametro - diametro_original

        print(f"\nJPEG Q={calidad}")
        print(f"Diametro: {diametro:.2f} px")
        print(f"Diferencia: {diferencia:+.2f} px")