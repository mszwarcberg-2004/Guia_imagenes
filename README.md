# Guía de imágenes: análisis de la respuesta pupilar en video

Trabajo práctico de procesamiento de imágenes biomédicas (PSIB). A partir de un video de un ojo con un estímulo luminoso, el pipeline extrae un frame y una región de interés (ROI), segmenta la pupila, calcula descriptores morfométricos y de textura, sigue su área y diámetro cuadro a cuadro y estima parámetros fisiológicos: latencia, amplitud y velocidades de constricción y redilatación. También incluye un análisis de compresión de imágenes (Huffman, RLE y JPEG).

Los resultados, las observaciones y las limitaciones del trabajo están en [`observaciones.md`](observaciones.md).

## Estructura del repositorio

```
Guia_imagenes/
├── datos/
│   ├── .gitkeep
│   └── Videos_ojo/             # Videos de entrada
├── imagenes/                   # Imágenes intermedias que el programa guarda y reutiliza
│   ├── frame.png
│   ├── roi.png
│   └── mascara_referencia.png
├── resultados/                 # Salidas del pipeline (CSV y máscara final)
│   └── .gitkeep
├── src/
│   ├── main.py                 # Pipeline completo: módulos 1 a 4
│   ├── funciones_modulo1.py    # VideoReader y procesamiento de imágenes
│   ├── funciones_modulo2.py    # Segmentación y morfología
│   ├── funciones_modulo3.py    # Descriptores y seguimiento temporal de la pupila
│   ├── funciones_modulo4.py    # Compresión: Huffman, RLE y JPEG
│   ├── visualizacion.py        # Funciones para mostrar imágenes e histogramas
│   ├── comparar_registros.py   # Procesa todos los videos y compara los registros
│   └── validar_mae_rmse.py     # Validación manual vs. automática (MAE y RMSE)
├── Guia_imagenes.ipynb         # Versión en notebook (ver nota abajo)
├── observaciones.md            # Resultados, observaciones y limitaciones
├── requirements.txt
├── .gitignore
└── README.md
```


El notebook `Guia_imagenes.ipynb` no se mantiene sincronizado con `src/`. Se conserva porque muestra las imágenes y los resultados intermedios en secuencia, lo que ayuda a seguir el trabajo.

## Módulos

| Módulo | Qué hace | Archivo |
|---|---|---|
| 1 | Extrae un frame del video y define la ROI. Estudia resolución espacial, cuantización, espacios de color, brillo y contraste, histogramas, ecualización, ruido y filtrado espacial, detección de bordes, Fourier, PSD, filtrado en frecuencia, PSNR y SSIM | `funciones_modulo1.py` |
| 2 | Segmenta la pupila con Otsu, K-Means, Snakes y Watershed, y los compara con una referencia manual mediante Dice. Aplica morfología, esqueletiza y evalúa con Dice e IoU. Guarda la máscara final | `funciones_modulo2.py` |
| 3 | Calcula descriptores morfométricos y de textura (primer orden y GLCM) de la pupila y el iris. Sigue la pupila en todo el video y genera la curva temporal, los parámetros fisiológicos y el control de artefactos | `funciones_modulo3.py` |
| 4 | Compara Huffman, RLE y JPEG (Q = 90, 50, 10 y 5) sobre la ROI, la máscara y el frame, con ratio, PSNR y SSIM. Mide cómo cambia el diámetro pupilar tras la compresión | `funciones_modulo4.py` |

Los demás archivos de `src/` son de apoyo:

| Archivo | Contenido |
|---|---|
| `visualizacion.py` | `mostrar_figura`, `mostrar_imagenes` (grilla de imágenes, con `vmin` y `vmax` únicos o por imagen) y `mostrar_histogramas` |
| `comparar_registros.py` | Procesa todos los videos de `datos/Videos_ojo/` con una ROI automática y compara latencia y amplitud entre registros |
| `validar_mae_rmse.py` | Compara mediciones automáticas con contornos dibujados a mano en 6 frames y calcula MAE y RMSE |

## Instalación

```bash
git clone https://github.com/mszwarcberg-2004/Guia_imagenes.git
cd Guia_imagenes

python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Las librerías principales son OpenCV, NumPy, Matplotlib, pandas y scikit-image. La textura GLCM usa `graycoprops`, que existe desde scikit-image 0.19.

## Uso

### Ejecutar el pipeline completo

1. Copiar el video a `datos/Videos_ojo/`.
2. Editar `VIDEO_PATH` en el bloque de constantes de `src/main.py`:

   ```python
   VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "NOMBRE_VIDEO.EXTENSIÓN"
   ```

3. Ejecutar **desde la raíz del repositorio**, porque `main.py` arma las rutas con la carpeta de trabajo:

   ```bash
   python src/main.py
   ```

`main.py` corre los cuatro módulos en orden y muestra los pasos intermedios (imágenes, histogramas y tablas). Cada figura detiene la ejecución hasta que se cierra la ventana.

### Interacción en la primera ejecución

Si no existen en `imagenes/`, el programa pide:

- **La ROI:** arrastrar un rectángulo sobre el frame y presionar Enter. Se guarda en `imagenes/roi.png`.
- **La máscara de referencia:** hacer clic alrededor del borde de la pupila y presionar Enter. Se guarda en `imagenes/mascara_referencia.png`.

En las ejecuciones siguientes se reutilizan esos archivos. Estos pasos abren ventanas, así que necesitan un entorno con pantalla.

El frame representativo se puede elegir modificando la constante `SEGUNDO`.

### Procesar un video nuevo

1. Copiar el video a `datos/Videos_ojo/` y cambiar `VIDEO_PATH`.
2. **Borrar `imagenes/roi.png` y `imagenes/mascara_referencia.png`** si el video es de otro registro. Si existen, se reutilizan y los análisis de los módulos 1, 2 y 4 se calculan sobre la ROI anterior. El seguimiento temporal del módulo 3 y sus CSV se calculan directamente sobre el video indicado.
3. Ejecutar `python src/main.py` y marcar la ROI y la referencia cuando se pidan.

Los CSV llevan el nombre del video, por lo que no se pisan entre registros distintos.

### Comparar todos los registros

```bash
python3 src/comparar_registros.py
```

Procesa cada video de `datos/Videos_ojo/` con una ROI automática propia, sin tocar `main.py` ni las imágenes guardadas, y no abre ventanas. Usa Watershed + dilatación para todos los videos, sin reevaluarlos. Genera `comparacion_registros.csv`, `variabilidad_registros.csv`, `errores_registros.csv` y un `log_<video>.txt` por video.

### Validar contra mediciones manuales

```bash
python src/validar_mae_rmse.py
```

Requiere haber ejecutado antes `main.py`, porque lee `resultados/mediciones_video5.csv`. Pide marcar la pupila a mano en 6 frames y guarda `validacion_mae_rmse_video5.csv`. El video y el CSV están definidos al comienzo del archivo (`VIDEO_PATH`, `CSV_PATH`).

## Salida estandarizada

Para cada video, el módulo 3 escribe en `resultados/` tres archivos CSV:

| Archivo | Contenido |
|---|---|
| `mediciones_<video>.csv` | Una fila por frame: `frame`, `tiempo_s`, `area_px2`, `diametro_px`, `centro_x`, `centro_y` y `detectada` (1 si hubo detección válida) |
| `serie_procesada_<video>.csv` | Curva temporal: `tiempo_s`, `area_original_px2`, `area_procesada_px2`, `diametro_original_px` y `diametro_procesado_px` |
| `parametros_fisiologicos_<video>.csv` | Una fila con el `estado` del cálculo, el área basal, `amplitud_px2`, `latencia_s`, `velocidad_constriccion_px2_s` y `velocidad_redilatacion_px2_s`, entre otros |

Los valores faltantes se representan con `NaN`. El área se mide en píxeles cuadrados y el diámetro es el equivalente de un círculo con esa área, `D = 2·√(A/π)`; no están calibrados en milímetros.

Además, el módulo 2 guarda `resultados/mascara_pupila_final.png`, y `main.py` guarda `imagenes/frame.png` y `imagenes/roi.png`.

## Cómo está organizado el código

```
video ─► frame y ROI (módulo 1) ─► máscara final de la pupila (módulo 2) ─► descriptores y seguimiento (módulo 3)
                                                                          └─► compresión (módulo 4)
```

- Cada módulo es una función en `main.py` (`modulo1`, `modulo2`, `modulo3`) que recibe lo que necesita por parámetro y devuelve lo que usa el siguiente. El módulo 1 devuelve `frame`, `roi` y `roi_gris`; el 2 devuelve `mascara_final`, `mejor_metodo` y `mejor_operacion`.
- Los cálculos reutilizables están en `funciones_modulo*.py`. `main.py` conserva el flujo, los `print` y los gráficos.
- Los videos se abren con `VideoReader`, que se usa con `with` y libera el archivo al terminar.

## Autores

**[Milena Szwarcberg, Julieta Mankevich y Emanuel Gallardo]**