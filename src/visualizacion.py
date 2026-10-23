#Este archivo contiene funciones útiles para la visualización
#de imágenes e histogramas en varios de los módulos

#Imports
import cv2
import matplotlib.pyplot as plt

#Función de graficar una imagen
def mostrar_figura(size ,titulo, imagen):
    plt.figure(figsize=size)
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

#Función para graficar varias imágenes en una grid
def mostrar_imagenes(imagenes, titulos, filas=1, figsize=None,
                     cmap="gray", vmin=None, vmax=None, interpolation=None):
    n = len(imagenes)
    columnas = -(-n // filas)  # division hacia arriba
    if figsize is None:
        figsize = (4 * columnas, 4 * filas)

    # Acepta un valor para todas las imagenes o una lista con uno por imagen
    vmins = vmin if isinstance(vmin, (list, tuple)) else [vmin] * n
    vmaxs = vmax if isinstance(vmax, (list, tuple)) else [vmax] * n

    if len(vmins) != n or len(vmaxs) != n:
        raise ValueError(
            "vmin/vmax deben ser un valor unico o una lista con un valor por imagen"
        )

    fig, ax = plt.subplots(filas, columnas, figsize=figsize)
    ax = ax.ravel() if n > 1 else [ax]

    for a, img, titulo, v0, v1 in zip(ax, imagenes, titulos, vmins, vmaxs):
        a.imshow(img, cmap=cmap, vmin=v0, vmax=v1,
                 interpolation=interpolation)
        a.set_title(titulo, fontsize=10)
        a.axis("off")
    for a in ax[n:]:  # oculta ejes sobrantes
        a.axis("off")

    plt.tight_layout()
    plt.show()

#Función para mostrar lo histogramas en escala de grises
def mostrar_histogramas(imagenes, titulos, log=False, figsize=None):

    n = len(imagenes)
    fig, ax = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    ax = [ax] if n == 1 else ax

    for a, img, titulo in zip(ax, imagenes, titulos):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        a.plot(hist)
        if log:
            a.set_yscale("log")
        a.set_title(titulo)
        a.set_xlabel("Intensidad")
        a.set_ylabel("Cantidad de pixeles")
        a.set_xlim([0, 255])

    plt.tight_layout()
    plt.show()