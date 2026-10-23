# Guia_imagenes
# Validación externa — Guía de Imágenes Biomédicas (Módulos 3 y 4)

Este programa independiente utiliza los algoritmos del repositorio y permite probar un **video nuevo, sin selección manual de ROI ni contorno**, dejando el TP principal sin modificar.

## Instalación

1. Clonar o descargar el repositorio completo.
2. Colocar `probar_video_externo.py` dentro de `src/` (al mismo nivel que `funciones_modulo3.py`, `funciones_modulo4.py` y `comparar_registros.py`).
3. Instalar las dependencias desde la raíz del proyecto: `python3 -m pip install -r requirements.txt`.
4. Ubicar el video de prueba, por ejemplo, en `datos/Videos_ojo/video_prueba.mp4`.
5. En `src/probar_video_externo.py`, modificar **solamente** `VIDEO_PATH` para señalar el archivo. Acepta una ruta relativa construida con `RAIZ` o una ruta absoluta mediante `Path("...")`.
6. Desde la carpeta raíz del repositorio, ejecutar `python3 src/probar_video_externo.py`.

**No ejecutar `main.py` para esta prueba:** la versión de desarrollo de `main.py` reutiliza una ROI y máscara guardadas.

## Archivos generados

Dentro de `resultados_externos/<nombre_video>/`:

- `curva_pupilar.csv`: serie temporal por frame, con `tiempo_s`, `area_px2`, `diametro_px`, mediciones originales, detección y centro estimado. Los faltantes se conservan como valores vacíos / NaN.
- `parametros_fisiologicos.csv`: estimaciones automáticas de latencia, amplitud, velocidad de constricción y redilatación, junto con estado del cálculo.
- `curva_pupilar.png`: gráfico de área y diámetro en función del tiempo.
- `revision_pupila.mp4`: video con centro y **círculo de diámetro equivalente**, no es el borde real de la máscara de segmentación. Permite verificar errores visualmente.
- `roi_detectada.png`, `mascara_inicial.png`: inicialización automática específica del video, sin reutilizar datos previos.
- `modulo4_compresion.txt`: comparación de Huffman, RLE, JPEG y efecto de JPEG sobre diámetro para imágenes extraídas del propio video.
- `ejecucion.txt` y carpeta `resultados/`: archivos originales producidos por el Módulo 3.

## Criterios y limitaciones

- La elección **Watershed + dilatación** proviene de la validación con los datos de desarrollo; no se recalcula contra una máscara manual del video externo.
- El seguimiento y la detección del estímulo son automáticos, pero pueden fallar con parpadeos, movimientos, reflejos o iluminación atípica.
- Si se muestra `Estado: Estimulo no detectado` (u otro estado distinto de `Calculados`), **no** se deben interpretar latencia y velocidades como valores válidos.
- Obtener un CSV no garantiza precisión. Evaluar la curva y la vista previa antes de sacar conclusiones.
- No se garantiza funcionamiento para todos los videos no vistos; es necesario probarlo con videos distintos al de desarrollo.
- Este archivo facilita la evaluación externa de **Módulos 3 y 4**. **No convierte por sí solo `main.py` en un pipeline integral 0–4 sin interacción**, que es un requisito adicional de la consigna.
