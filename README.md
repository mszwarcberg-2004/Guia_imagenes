# Reflejo fotomotor pupilar: procesamiento de video

TP de Procesamiento de Señales e Imágenes Biomédicas (PSIB). A partir de un video del ojo,
el programa sigue la pupila frame a frame, arma la curva de área y diámetro y calcula
latencia, amplitud y velocidad de constricción y de redilatación.

El repositorio tiene dos partes:

1. **Prueba de un video nuevo** (`src/probar_video_externo.py`): procesa el video de la cátedra
   de forma automática y entrega la curva y los parámetros en **un único archivo**.
2. **Nuestro trabajo** (`src/main.py`): todos los ítems de la consigna (Módulos 1 a 4) sobre
   nuestro video, con la ROI y el contorno de referencia que marcamos a mano.

Las observaciones y resultados de cada módulo están en `observaciones.md`.

---

## Instalación (una sola vez)

Requiere Python 3.10 o superior (desarrollado con 3.14). Desde la carpeta raíz del repositorio:

```
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Todos los comandos se ejecutan desde la carpeta raíz del repositorio**, no desde `src/`.

---

## Para la cátedra: probar un video nuevo

Lo único que hay que cambiar es la ruta del video. No se marca nada a mano ni se abren ventanas.

**1.** Abrir `src/probar_video_externo.py` y, en la **línea 54**, reemplazar la ruta que está entre comillas
por la del video:

```python
VIDEO_PATH = "datos/Videos_ojo/video_prueba.mp4"      # <- línea actual
VIDEO_PATH = "/Users/nombre/Descargas/video.mov"      # <- así, con la ruta de su video
```

La ruta puede ser absoluta (el video puede estar en cualquier lugar de la computadora) o relativa a la
carpeta del repositorio. Esa línea está marcada en el archivo con un cartel que dice
"LO ÚNICO QUE HAY QUE CAMBIAR".

**2.** Ejecutar, desde la carpeta raíz del repositorio:

```
python3 src/probar_video_externo.py
```

*Alternativa sin editar el archivo:* pasar la ruta al ejecutar,
`python3 src/probar_video_externo.py ruta/al/video.mov` (si tiene espacios, entre comillas).

Tarda aproximadamente entre 30 y 60 segundos por video, según la computadora.

### Resultado

Todo queda en **`resultados_externos/<nombre_del_video>/`**:

| Archivo | Contenido |
|---|---|
| **`resultados.xlsx`** | **El resultado.** Un solo archivo con dos hojas: **`Parametros`** (los parámetros fisiológicos, cada uno una sola vez) y **`Curva`** (área y diámetro de la pupila frame a frame) |
| `curva_pupilar.png` | La misma curva, como gráfico |
| `extras/` | Material de apoyo: `revision_pupila.mp4` (el video con el círculo de la pupila detectada, para revisar el seguimiento), `ejecucion.txt` (log) y las tablas y figuras internas del Módulo 3 |

Cada video crea su propia carpeta, y si se vuelve a procesar el mismo video la carpeta se reemplaza por completo.
`resultados_externos/video_prueba/` es un ejemplo de salida, hecho con nuestro propio video de prueba.

*(Si en la computadora no está instalado `openpyxl`, en lugar del Excel se guardan `parametros.csv` y `curva.csv`
con la misma información; se instala con `pip install openpyxl`.)*

### Hoja `Parametros`

| Fila | Descripción |
|---|---|
| **Latencia** (s) | Tiempo entre el encendido de la luz y el inicio de la constricción |
| **Amplitud de constricción** (px² y % del área basal) | Cuánto se reduce el área de la pupila |
| **Velocidad de constricción** (px²/s) | Velocidad media: la amplitud dividida por el tiempo entre el inicio de la constricción y el área mínima |
| **Velocidad de redilatación** (px²/s) | Velocidad media desde el apagado de la luz: cambio de área dividido por el tiempo hasta el final de la ventana de medición (hasta 4 s). Solo si el video incluye el apagado |
| **Estado del cálculo** | Qué tan confiables son los valores (ver abajo) |
| *Datos del registro* | Video, método de segmentación, tipo de registro (`A`: la luz se enciende, `B`: se apaga), instantes de encendido y de apagado, área basal y mínima, frames totales y con pupila detectada |

Un valor que no se pudo calcular aparece como `no calculado`. Los valores están en píxeles, sin calibrar a milímetros. Las velocidades pico (máxima pendiente en 0,2 s) quedan en `extras/resultados/detalle/parametros_detallados_<video>.csv`.

### Hoja `Curva`

Una fila por frame:

| Columna | Descripción |
|---|---|
| `frame` | Número de frame (desde 0) |
| `tiempo_s` | Tiempo en segundos |
| `area_px2` | Área de la pupila (px²), ya limpiada de parpadeos y valores atípicos. Vacía donde no hay una medición confiable |
| `diametro_px` | Diámetro equivalente (px): 2·√(área/π) |
| `area_original_px2`, `diametro_original_px` | La medición de ese frame, sin limpiar |
| `detectada` | 1 si se detectó la pupila en ese frame, 0 si no |

### Cómo leer el estado

| Estado | Qué significa |
|---|---|
| `Constriccion: Calculada` (y `Redilatacion: Calculada` si hay apagado) | Valores completos |
| `... (latencia < 0.1 s: revisar)` | La latencia es menor que cualquier valor fisiológico: no usarla sin revisar la curva |
| `Parcial (amplitud comparable al ruido: sin latencia)` | Hay amplitud pero la señal es muy ruidosa para fijar el inicio; la latencia queda sin calcular |
| `Parcial (luz no detectada...)`, `Basal insuficiente`, `Respuesta incompleta`, `Sin constriccion`, `Sin redilatacion`, `Respuesta insuficiente` | Faltan datos o no hubo una respuesta medible; revisar el video |

---

## Nuestro trabajo: todos los ítems de la consigna (Módulos 1 a 4)

```
python3 src/main.py
```

Usa nuestro video de desarrollo (`datos/Videos_ojo/video5.mov`) y la ROI y el contorno de
referencia ya marcados a mano (`imagenes/roi.png` y `imagenes/mascara_referencia.png`), así que
no pide nada. Se abren ventanas con los gráficos de cada módulo (hay que cerrarlas una por una para
continuar) y el proceso tarda unos minutos.

- **Módulos 1 y 2:** análisis de la imagen y comparación de Otsu, K-Means, Snakes y Watershed contra el
  contorno manual (Dice); luego, operaciones morfológicas.
- **Módulo 3:** descriptores y textura del frame segmentado, y seguimiento de la pupila en todo el video
  (curva y parámetros fisiológicos).
- **Módulo 4:** compresión (Huffman, RLE, JPEG).

Los resultados quedan en `resultados/` (parámetros y curva de `video5`, gráficos en `figuras/` y
tablas de apoyo en `detalle/`).

> Si se cambia el video de desarrollo hay que borrar `imagenes/roi.png` y
> `imagenes/mascara_referencia.png` para volver a marcarlos a mano.

La validación manual de nuestro video (MAE y RMSE) se hace con `python3 src/validar_mae_rmse.py`.

---

## Método de segmentación

La prueba de la cátedra usa **Snakes sin morfología**, el método más robusto de nuestras pruebas: fue el de
mayor Dice contra el contorno manual (Snakes 0,976; Watershed 0,966; K-Means K=5 0,950; Otsu 0,785).
Queda fijo en las constantes `METODO` y `MORFOLOGIA` de `src/probar_video_externo.py` y no se vuelve a elegir
para el video de la cátedra, porque no hay un contorno manual contra el cual compararlo.

## Cómo funciona el seguimiento

1. Se estima automáticamente la posición y el radio de la pupila con 15 frames del video.
2. En cada frame se busca la pupila cerca de la anterior y se segmenta, con controles de
   calidad (área, forma, saltos entre frames).
3. Se mide la luminosidad alrededor del ojo para detectar cuándo se enciende y se apaga la luz.
4. Se limpia la serie (valores atípicos, parpadeos) y se calculan los parámetros: la constricción
   desde el encendido y la redilatación desde el apagado.

## Limitaciones

- Los parpadeos, movimientos, reflejos del flash o una iluminación atípica pueden degradar el
  seguimiento. Un CSV no garantiza precisión: conviene mirar `curva_pupilar.png` y
  `extras/revision_pupila.mp4`. Si un parpadeo o un destello coincide con el encendido, la latencia
  puede quedar sobrestimada.
- `probar_video_externo.py` cubre el Módulo 3 (seguimiento y parámetros) sobre un video nuevo. Los Módulos
  1, 2 y 4 se ejecutan en `main.py`, que requiere haber marcado a mano la ROI y el contorno de referencia
  sobre el video de desarrollo.
- La elección del método se hizo con un solo frame de referencia de nuestro video.

---

## Estructura

```
src/
  probar_video_externo.py    prueba automática de un video nuevo (para la cátedra)
  main.py                    nuestro trabajo: Módulos 1 a 4 sobre el video de desarrollo
  validar_mae_rmse.py        validación manual del equipo (MAE y RMSE)
  funciones_modulo1.py ... funciones_modulo4.py, visualizacion.py
datos/Videos_ojo/            videos grabados por el equipo
imagenes/                    ROI y contorno de referencia de nuestro video
resultados/                  salidas de main.py
resultados_externos/         salidas de probar_video_externo.py (una carpeta por video)
observaciones.md             resultados y observaciones por módulo
requirements.txt
```
