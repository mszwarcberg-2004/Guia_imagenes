
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

**Observaciones:** Pendiente.

## Módulo 2. Segmentación y morfología matemática

**Observaciones:** Pendiente.

## Módulo 3. Extracción de características y dinámica temporal

**Observaciones:** Pendiente.

## Módulo 4. Compresión de imágenes

**Observaciones:** Pendiente.
