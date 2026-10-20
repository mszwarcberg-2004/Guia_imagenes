#Imports
import cv2 #cv2: librería que permite abrir videos, extraer frames y procesar imagenes.
import matplotlib.pyplot as plt
from pathlib import Path

#Módulo 1:  

# Ruta del video
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
VIDEO_PATH = CARPETA_PROYECTO / "datos" / "Videos_ojo" / "video1.mov"
#__file__ es la ubicacion del archivo que estamos trabajando, en este caso main.py

# Abrir el video
video = cv2.VideoCapture(str(VIDEO_PATH))

#Si el video no se puede abrir, tira error
if not video.isOpened():
    raise ValueError("No se pudo abrir el video")

# Obtener la cantidad de frames
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Seleccionar el frame de la mitad del video
frame_num = total_frames // 2
video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

# Extraer el frame
ret, frame = video.read()
video.release()

if not ret:
    raise ValueError("No se pudo extraer el frame")

# Convertir de BGR (en lo que trabaja opencvs) a RGB para visualizarlo
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mostrar el frame
plt.figure(figsize=(8, 6))
plt.imshow(frame_rgb)
plt.title("Frame representativo del video")
plt.axis("off")
plt.show()

# Seleccionar manualmente la region del ojo
x, y, w, h = cv2.selectROI("Seleccionar ROI", frame, False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise ValueError("No se selecciono una ROI")

# Recortar el ROI seleccionado de la imagen
roi = frame[y:y+h, x:x+w]

# Crear carpeta IMAGENES, donde vamos a guardar el frame y el roi
CARPETA_IMAGENES = CARPETA_PROYECTO / "imagenes"
CARPETA_IMAGENES.mkdir(exist_ok=True)

# Guardar frame completo y ROI como pngs en mi compu
cv2.imwrite(str(CARPETA_IMAGENES / "frame.png"), frame)
cv2.imwrite(str(CARPETA_IMAGENES / "roi.png"), roi)

# Visualizar ROI
roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(roi_rgb)
plt.title("Region de interes (ROI)")
plt.axis("off")
plt.show()
