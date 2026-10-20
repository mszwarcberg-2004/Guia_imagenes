import cv2 #cv2: librería que permite abrir videos, extraer frames y procesar imagenes.
import matplotlib.pyplot as plt


#Módulo 1:  

# Ruta del video
VIDEO_PATH = "datos/Videos_ojo/video1.mov"

# Abrir el video
video = cv2.VideoCapture(VIDEO_PATH)

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

# Convertir de BGR a RGB para visualizarlo
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mostrar el frame
plt.figure(figsize=(8, 6))
plt.imshow(frame_rgb)
plt.title("Frame representativo del video")
plt.axis("off")
plt.show()
