"""Procesa videos del TP y compara latencia/amplitud entre registros.

Ejecutar desde la raiz: python3 src/comparar_registros.py
No modifica main.py ni las ROI/mascaras existentes.
"""
from pathlib import Path
import contextlib
import io
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No abrir decenas de ventanas al procesar varios videos
import matplotlib.pyplot as plt

from funciones_modulo3 import modulo3

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_VIDEOS = RAIZ / 'datos' / 'Videos_ojo'
RESULTADOS = RAIZ / 'resultados'
EXTENSIONES = {'.mp4', '.mov', '.avi', '.mkv'}

# Metodo y morfologia previamente seleccionados con la referencia manual
# del video de desarrollo. NO son reevaluados automaticamente para cada video.
METODO = 'Watershed'
MORFOLOGIA = 'Dilatacion'


def preparar_roi_automatica(ruta):
    """Busca una pupila visible para aportar una ROI propia de cada video a 3.1.

    Es una inicializacion automatica aproximada; NO reemplaza una referencia manual.
    Si no encuentra un circulo convincente, falla en lugar de reutilizar otra ROI.
    """
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise ValueError('No se pudo abrir el video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError('FPS invalido')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Probar frames de varios momentos; evitar el medio del estimulo si es posible
    segundos = [0.5, 1.0, 0.1, 2.5, 4.0]
    try:
        for s in segundos:
            indice = min(max(0, int(s * fps)), max(0, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, frame = cap.read()
            if not ok:
                continue
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            alto, ancho = gris.shape
            suavizada = cv2.GaussianBlur(gris, (7, 7), 1.5)
            circulos = cv2.HoughCircles(
                suavizada, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=25, param1=100, param2=22,
                minRadius=max(5, round(ancho * .010)),
                maxRadius=max(10, round(ancho * .085))
            )
            if circulos is None:
                continue
            opciones = []
            yy, xx = np.ogrid[:alto, :ancho]
            for cx, cy, r in circulos[0]:
                # Restringir busqueda al area central, evitando bordes del rostro
                if not (.12 * ancho < cx < .88 * ancho and .15 * alto < cy < .85 * alto):
                    continue
                d2 = (xx - cx)**2 + (yy - cy)**2
                interior = gris[d2 <= (.75*r)**2]
                anillo = gris[(d2 >= (1.25*r)**2) & (d2 <= (1.7*r)**2)]
                if interior.size < 10 or anillo.size < 10:
                    continue
                contraste = float(np.median(anillo) - np.median(interior))
                if contraste < 12:
                    continue
                opciones.append((contraste, float(cx), float(cy), float(r)))
            if not opciones:
                continue
            _, cx, cy, r = max(opciones)
            # ROI local amplia, incluye la pupila y un margen de iris
            m = int(round(2.0 * r))
            x1, x2 = max(0, round(cx) - m), min(ancho, round(cx) + m + 1)
            y1, y2 = max(0, round(cy) - m), min(alto, round(cy) + m + 1)
            roi = frame[y1:y2, x1:x2].copy()
            roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Mascara inicial por intensidad dentro de un disco alrededor del centro
            yc, xc = cy - y1, cx - x1
            y, x = np.ogrid[:roi_gris.shape[0], :roi_gris.shape[1]]
            dist = np.hypot(x - xc, y - yc)
            zona = dist <= 1.35 * r
            valores = roi_gris[zona]
            if len(valores) < 20:
                continue
            umbral, _ = cv2.threshold(valores.reshape(-1, 1), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mascara = ((roi_gris <= umbral) & zona).astype(np.uint8)
            n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(mascara, 8)
            if n <= 1:
                continue
            elegible = min(range(1, n), key=lambda i: np.hypot(centros[i, 0]-xc, centros[i, 1]-yc))
            mascara = (etiquetas == elegible).astype(np.uint8) * 255
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            area = np.count_nonzero(mascara)
            if not (.25*np.pi*r*r <= area <= 2.5*np.pi*r*r):
                continue
            return frame, roi, roi_gris, mascara, s
    finally:
        cap.release()
    raise ValueError('No se encontro una pupila inicial confiable para crear ROI automatica')


def main():
    RESULTADOS.mkdir(exist_ok=True)
    videos = sorted(p for p in CARPETA_VIDEOS.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not videos:
        raise FileNotFoundError(f'No hay videos en {CARPETA_VIDEOS}')

    # Para que el lote no se detenga al llegar a una figura.
    plt.show = lambda *args, **kwargs: plt.close('all')

    filas, errores = [], []
    for video in videos:
        print(f'\n========== {video.name} ==========')
        try:
            frame, roi, roi_gris, mascara, segundo_roi = preparar_roi_automatica(video)
            print(f'ROI automatica localizada cerca de {segundo_roi:.1f} s')
            print('Procesando seguimiento temporal...')
            # El modulo original imprime resultados extensos: conservarlos en un log individual.
            log = io.StringIO()
            with contextlib.redirect_stdout(log), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                modulo3(frame, roi, roi_gris, mascara, METODO, MORFOLOGIA, video, RAIZ)
            (RESULTADOS / f'log_{video.stem}.txt').write_text(log.getvalue(), encoding='utf-8')
            ruta = RESULTADOS / f'parametros_fisiologicos_{video.stem}.csv'
            df = pd.read_csv(ruta)
            if df.empty:
                raise ValueError('CSV de parametros vacio')
            fila = df.iloc[0].to_dict()
            fila['estado'] = str(fila.get('estado', ''))
            basal = pd.to_numeric(fila.get('area_basal_px2'), errors='coerce')
            amplitud = pd.to_numeric(fila.get('amplitud_px2'), errors='coerce')
            fila['amplitud_relativa_pct'] = (100*amplitud/basal if pd.notna(basal) and basal > 0 else np.nan)
            filas.append(fila)
            print(f"Estado: {fila['estado']} | Latencia: {fila.get('latencia_s')} s | Amplitud: {amplitud} px²")
        except Exception as exc:
            errores.append({'video': video.name, 'error': str(exc)})
            print(f'No se pudo analizar: {exc}')
        finally:
            plt.close('all')

    if errores:
        pd.DataFrame(errores).to_csv(RESULTADOS / 'errores_registros.csv', index=False)
    if not filas:
        print('\nNingun registro produjo parametros. Revisar errores_registros.csv.')
        return

    tabla = pd.DataFrame(filas)
    for col in ['latencia_s', 'amplitud_px2', 'amplitud_relativa_pct']:
        tabla[col] = pd.to_numeric(tabla[col], errors='coerce')
    tabla.to_csv(RESULTADOS / 'comparacion_registros.csv', index=False)

    # No mezclar ensayos no comparables: comprobar en el protocolo cuáles son
    # oscuridad->luz y cuáles luz->oscuridad. Esta tabla es exploratoria.
    validos = tabla[(tabla['estado'] == 'Calculados') &
                    tabla['latencia_s'].notna() &
                    tabla['amplitud_relativa_pct'].notna()].copy()

    print('\n--- RESULTADOS POR VIDEO ---')
    print(tabla[['video', 'estado', 'latencia_s', 'amplitud_px2', 'amplitud_relativa_pct']].to_string(index=False))
    if len(validos) < 2:
        print('\nNo hay 2 registros con parametros calculados: aun no se puede estimar variabilidad.')
        return

    resumen = validos[['latencia_s', 'amplitud_px2', 'amplitud_relativa_pct']].agg(['count', 'mean', 'std', 'min', 'max'])
    resumen.to_csv(RESULTADOS / 'variabilidad_registros.csv')
    print('\n--- VARIABILIDAD (SOLO REGISTROS CON ESTADO CALCULADOS) ---')
    print(resumen.round(3).to_string())
    print('\nArchivos: resultados/comparacion_registros.csv y resultados/variabilidad_registros.csv')
    print('ATENCION: revisar visualmente cada video, los picos por parpadeo y el tipo de estimulo antes de informar resultados finales.')


if __name__ == '__main__':
    main()
