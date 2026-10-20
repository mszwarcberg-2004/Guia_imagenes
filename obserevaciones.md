
# Observaciones y resultados — TP Imágenes Biomédicas

## Módulo 1. Procesamiento de imágenes

#### 1.2.1 Histograma, brillo y contraste

Se calculó el histograma del frame completo en escala de grises y se aplicaron dos transformaciones: aumento de brillo mediante la suma de 50 unidades y modificación del contraste mediante la multiplicación de las intensidades por 1,5.

En ambos casos, el histograma se desplazó hacia intensidades mayores y se observa mucha saturación, debido a que numerosos píxeles alcanzaron el valor máximo de 255, provocando pérdida de información.

#### 1.2.2 Escalado lineal por rango de interés

Se seleccionaron automáticamente los límites del intervalo utilizando los porcentajes 5 y 95% de las intensidades de la ROI en escala de grises, para que el intervalo seleccionado contenga aproximadamente el 90% central de los valores de intensidad, excluyendo los extremos de la distribución.

Se obtuvieron los siguientes valores:

- f1 = 33
- f2 = 99

El intervalo seleccionado se expandió linealmente al rango de 0 a 255, aumentando la separación entre intensidades de la pupila y el iris. A los valores menores a f1 se les asignó 0 y a los mayores a f2, 255.

Sin embargo, se observó saturación en algunas regiones del iris, lo que produjo pérdida de detalles de textura.

#### 1.2.3 Ecualización global y CLAHE

Se aplicaron dos métodos de ecualización sobre la ROI en escala de grises:

- **Ecualización global:** redistribuye las intensidades utilizando el histograma de toda la imagen, aplicando la misma transformación a todos los píxeles.
- **CLAHE:** divide la imagen en regiones pequeñas y realiza una ecualización local, limitando la amplificación del contraste para reducir la aparición de ruido y artefactos.

**Observaciones:**

- **Escalado lineal:** aumenta la separación de intensidades entre la pupila y el iris, pero produce saturación y pérdida de detalles en algunas regiones.
- **Ecualización global:** mejora considerablemente el contraste y distribuye las intensidades en un rango más amplio, aunque también genera zonas muy claras y puede exagerar algunas texturas.
- **CLAHE:** produce una mejora más moderada del contraste, conservando mejor los detalles del iris y evitando una saturación tan marcada.

En esta ROI, **CLAHE presenta el resultado visual más equilibrado**, ya que mejora el contraste local sin generar cambios tan bruscos de intensidad. Sin embargo, la imagen sigue presentando reflejos luminosos y cierta falta de definición.

Con respecto al ruido, no se observa un ruido muy marcado, aunque la imagen tiene poca resolución y algunos reflejos luminosos. Eso no significa que no tenga ruido, sino que no podemos distinguirlo claramente a simple vista.
#### 1.2.4 Ruido y filtrado espacial

**Observaciones:** Pendiente.

### 1.3 Detección de bordes y dominio frecuencial

#### 1.3.1 Detección de bordes

Se aplicaron los operadores Roberts, Sobel, Prewitt, Laplaciano y Canny sobre la ROI original y sobre la ROI filtrada mediante un filtro gaussiano de 3×3, para comparar la detección y continuidad del contorno pupilar.

**Observaciones:**

- **Roberts:** permite identificar el contorno de la pupila, aunque también resalta el reflejo luminoso y algunas variaciones de intensidad del iris.
- **Sobel y Prewitt:** presentan resultados similares, detectando claramente el contorno pupilar, aunque con bordes relativamente gruesos.
- **Laplaciano:** resalta variaciones bruscas de intensidad, pero el contorno de la pupila aparece menos definido y se detectan detalles adicionales.
- **Canny:** produce el contorno más fino y prácticamente continuo, aunque también identifica bordes correspondientes a los reflejos luminosos.

Al comparar la imagen original con la filtrada, se observa que el filtro gaussiano suaviza algunas variaciones de intensidad y reduce detalles pequeños. Sin embargo, no produce una mejora significativa en la continuidad del contorno pupilar.

En esta ROI, **Canny presenta el resultado más conveniente para identificar el borde de la pupila**, ya que genera un contorno delgado y prácticamente cerrado. Además, el filtrado gaussiano no resulta indispensable para obtener este resultado, aunque puede ser útil para atenuar pequeñas variaciones de intensidad.

#### 1.3.2 Transformada de Fourier 2D

Se calculó la Transformada de Fourier 2D de la ROI original y se representaron su magnitud en escala logarítmica y su fase. Luego, se reconstruyó la imagen mediante la transformada inversa utilizando únicamente la magnitud, únicamente la fase y combinando magnitudes y fases de la ROI original y la filtrada con gaussiano.

**Observaciones:**

- **Magnitud:** presenta una mayor concentración de valores en la región central del espectro, lo que indica un predominio de bajas frecuencias espaciales, asociadas a variaciones suaves de intensidad.
- **Fase:** presenta una distribución visualmente desordenada, pero contiene información importante sobre la organización espacial de las estructuras.
- **Reconstrucción con solo magnitud:** se pierde gran parte de la estructura reconocible de la imagen, por lo que la pupila deja de distinguirse claramente.
- **Reconstrucción con solo fase:** se conservan algunos contornos y características estructurales, aunque se pierde la apariencia original de las intensidades.
- **Reconstrucciones cruzadas:** al combinar la magnitud de una imagen con la fase de la otra, se obtienen resultados muy similares a la ROI original, debido a que el filtrado gaussiano no modifica considerablemente la organización espacial de la imagen.

Por lo tanto, **la fase resulta fundamental para conservar la información estructural y la ubicación de los elementos de la imagen**, mientras que la magnitud aporta información sobre la amplitud de las componentes frecuenciales.

#### 1.3.3 Densidad espectral de potencia (PSD)

Se calculó la densidad espectral de potencia como el cuadrado de la magnitud de la Transformada de Fourier, comparando la ROI original con la ROI filtrada mediante gaussiano.

**Observaciones:**

En ambas imágenes se observa una mayor concentración de potencia en el centro del espectro, correspondiente a las bajas frecuencias espaciales.

Después del filtrado gaussiano, las regiones alejadas del centro presentan una menor intensidad, lo que indica una disminución de las componentes de alta frecuencia.

Esto es consistente con el efecto suavizante del filtro gaussiano, que atenúa variaciones rápidas de intensidad asociadas tanto al ruido como a los detalles finos de la imagen.

Sin embargo, **la reducción de altas frecuencias no implica necesariamente una mejor detección de la pupila**, ya que sus bordes también contienen información de alta frecuencia.

#### 1.3.4 Filtrado en el dominio frecuencial

Se diseñaron filtros pasa-bajos y pasa-altos ideales en el dominio frecuencial y se reconstruyó el frame completo mediante la Transformada Inversa de Fourier. Los resultados se compararon con el filtrado gaussiano espacial.

**Observaciones:**

- **Filtro pasa-bajos frecuencial:** produce un suavizado general de la imagen, disminuyendo la definición de detalles finos como las pestañas y las texturas del iris.
- **Filtro pasa-altos frecuencial:** resalta los bordes y las transiciones bruscas de intensidad, incluyendo los contornos del ojo, las pestañas y la pupila.
- **Filtro gaussiano espacial:** genera un suavizado más moderado que el pasa-bajos frecuencial utilizado, conservando mejor la definición general de las estructuras.

En las máscaras frecuenciales, el filtro pasa-bajos conserva las frecuencias cercanas al centro del espectro, mientras que el pasa-altos conserva las frecuencias alejadas del centro.

También se observan algunas oscilaciones alrededor de bordes de alto contraste en el resultado del pasa-bajos, compatibles con el efecto *ringing* característico de los filtros ideales.

En esta comparación, **el filtro gaussiano espacial conserva mejor los detalles de la imagen que el pasa-bajos frecuencial implementado**, aunque este resultado depende de los parámetros seleccionados para cada filtro.

#### Conclusiones generales del apartado 1.3

A partir de las comparaciones realizadas, se observó que **Canny permite identificar de manera más clara y continua el contorno pupilar**, tanto en la ROI original como en la filtrada.

El filtro gaussiano permite suavizar pequeñas variaciones de intensidad, pero no elimina los reflejos luminosos del flash, que continúan generando bordes adicionales. Por lo tanto, no se identificó un filtro que permita simultáneamente preservar por completo el contorno pupilar y eliminar dichos reflejos.

Por otra parte, el análisis de Fourier y de la PSD permitió comprobar que el filtrado modifica la distribución de las componentes frecuenciales y que la fase contiene información fundamental sobre la estructura espacial de la imagen.

En conjunto, los resultados muestran que **no siempre es conveniente aplicar un filtrado previo**, ya que la reducción de ruido o detalles también puede afectar los bordes que se desean conservar.

## Módulo 2. Segmentación y morfología matemática

**Observaciones:** Pendiente.

## Módulo 3. Extracción de características y dinámica temporal

**Observaciones:** Pendiente.

## Módulo 4. Compresión de imágenes

**Observaciones:** Pendiente.
