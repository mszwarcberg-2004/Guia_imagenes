
# Observaciones y resultados — TP Imágenes Biomédicas

## Módulo 1. Procesamiento de imágenes
### 1.1 Carga, muestreo y espacios de color

#### 1.1.1 Extracción del frame y selección de la ROI

Se extrajo un frame representativo del video registrado y se seleccionó una región de interés (ROI) centrada en la pupila y parte del iris, para trabajar específicamente sobre las estructuras que se desean analizar.

**Observaciones:**

En el frame completo se distingue claramente el ojo, mientras que la ROI permite observar con mayor detalle la pupila y su contorno.

Sin embargo, se identifica un reflejo luminoso sobre el borde izquierdo de la pupila, que podría interferir en las etapas posteriores de detección de bordes y segmentación. Además, la ROI presenta una resolución relativamente baja, lo que limita la definición de algunos detalles.

#### 1.1.2 Dimensiones y profundidad de bits

Se analizaron las dimensiones, la cantidad de canales y la profundidad de bits de la ROI original.

Se obtuvo una imagen de **85 × 56 píxeles**, con tres canales de color (RGB) y una profundidad de **8 bits por canal**, equivalente a 24 bits por píxel.

Estas características permiten representar 256 niveles de intensidad por canal, aunque la resolución espacial limitada puede dificultar la delimitación precisa de la pupila.

#### 1.1.3 Resolución espacial: submuestreo

Se redujo la resolución espacial de la ROI al 50%, 25% y 5% de sus dimensiones originales, con el objetivo de evaluar la pérdida de definición.

**Observaciones:**

- **Resolución original (85 × 56):** permite distinguir la pupila, su contorno y algunas características del iris, aunque la imagen ya presenta cierta pixelación.
- **Resolución al 50% (42 × 28):** la pupila sigue siendo reconocible, pero se pierden detalles finos y disminuye la definición de los bordes.
- **Resolución al 25% (21 × 14):** aumenta considerablemente la pixelación, el contorno pupilar se vuelve más irregular y se pierde gran parte de la textura del iris.
- **Resolución al 5% (4 × 3):** la imagen queda reducida a unos pocos píxeles y prácticamente desaparece la información necesaria para distinguir las estructuras oculares.

Por lo tanto, **a medida que disminuye la resolución espacial, se pierde información sobre la forma y los detalles de la pupila**. Esto es especialmente importante para el procesamiento posterior, ya que una resolución insuficiente puede afectar la precisión de las mediciones de área y diámetro pupilar.

#### 1.1.4 Cuantización: profundidad de bits

Se redujo la profundidad de bits de la ROI original de 8 bits a 4 bits y 2 bits por canal, disminuyendo la cantidad de niveles de intensidad disponibles.

**Observaciones:**

- **8 bits:** la imagen presenta transiciones de intensidad relativamente suaves y permite distinguir la pupila del iris.
- **4 bits:** se observan cambios más bruscos entre tonalidades y una pérdida moderada de detalles, aunque la pupila continúa siendo reconocible.
- **2 bits:** aparece una degradación considerable de los colores y las intensidades, con regiones más uniformes y transiciones abruptas entre niveles. Se pierde gran parte de la información de textura.

A diferencia del submuestreo espacial, la cuantización no reduce la cantidad de píxeles, sino la cantidad de intensidades que puede representar cada uno.

En este caso, **la reducción a 2 bits genera una pérdida importante de información visual**, aunque la forma general de la pupila todavía puede distinguirse.

#### 1.1.5 Espacios de color

Se separó la ROI en sus tres canales RGB y se realizó la conversión al espacio HSV, visualizando las componentes H (Matiz), S (Saturación) y V (Valor) de manera independiente.

**Observaciones:**

**Canales RGB:**

Los tres canales permiten identificar la pupila como una región oscura y distinguirla del iris. Sin embargo, presentan diferencias de intensidad y contraste según la componente de color analizada.

En las imágenes obtenidas, las diferencias entre los canales no son demasiado pronunciadas y el reflejo luminoso aparece claramente en los tres.

**Espacio HSV:**

- **H (Matiz):** representa las diferencias de tonalidad. Se observan variaciones entre la región pupilar y el iris, aunque el resultado presenta zonas con cambios bruscos debido a la baja saturación de algunos píxeles.
- **S (Saturación):** representa la intensidad o pureza del color y permite distinguir variaciones de saturación entre distintas regiones del iris y la pupila.
- **V (Valor):** representa el brillo de los píxeles. La pupila se observa como una región oscura, mientras que el reflejo luminoso presenta una intensidad elevada.

En esta ROI, **la componente V permite distinguir visualmente la pupila del iris de manera más clara**, ya que conserva las diferencias de luminosidad entre ambas estructuras.

Sin embargo, el reflejo sigue apareciendo como una región muy brillante, por lo que la conversión de espacio de color no elimina este artefacto.

#### Conclusiones generales del apartado 1.1

A partir de las comparaciones realizadas, se observó que tanto la resolución espacial como la profundidad de bits influyen en la cantidad de información disponible para analizar la pupila.

El submuestreo produjo una pérdida progresiva de definición, mientras que la reducción de la cuantización generó transiciones de intensidad más abruptas y una disminución de los detalles de textura.

Por otra parte, la comparación de los espacios de color permitió identificar diferencias en la representación de las estructuras oculares. En particular, la componente V del espacio HSV resultó útil para distinguir las regiones oscuras de la pupila respecto del iris.

En conjunto, estos resultados muestran la importancia de **conservar una resolución y profundidad de bits suficientes para preservar el contorno pupilar**, especialmente considerando que la ROI original ya presenta una resolución limitada y reflejos luminosos que pueden dificultar su procesamiento posterior.

#### 1.2.1 Histograma, brillo y contraste

Se calculó el histograma del frame completo en escala de grises y se aplicaron dos transformaciones: aumento de brillo mediante la suma de 50 unidades y modificación del contraste mediante la multiplicación de las intensidades por 1,5.

En ambos casos, el histograma se desplazó hacia intensidades mayores y se observa mucha saturación, debido a que numerosos píxeles alcanzaron el valor máximo de 255, provocando pérdida de información.

#### 1.2.2 Escalado lineal por rango de interés

Se seleccionaron automáticamente los límites del intervalo utilizando los porcentajes 5 y 95% de las intensidades de la ROI en escala de grises, para que el intervalo seleccionado contenga aproximadamente el 90% central de los valores de intensidad, excluyendo los extremos de la distribución.

Se obtuvieron los siguientes valores:

- f1 = 34
- f2 = 98

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

### 2.1 Segmentación de la pupila — ROI original

Se aplicaron los métodos Otsu, K-Means, Snakes y Watershed sobre la ROI original en escala de grises, sin filtrado adicional.

- **Otsu:** calculó un umbral de 62 y permitió identificar la región oscura correspondiente a la pupila. Sin embargo, también segmentó otras zonas oscuras de la ROI y dejó pequeños agujeros asociados a los reflejos.
- **K-Means:** se evaluaron valores de K entre 2 y 5. Al aumentar K, disminuyeron las regiones seleccionadas fuera de la pupila. Las máscaras con K=3, K=4 y K=5 resultaron similares, aunque conservaron pequeñas imperfecciones.
- **Snakes:** el contorno inicial se ubicó alrededor de la pupila, pero durante la evolución se contrajo hacia el reflejo luminoso, por lo que no logró delimitar correctamente el borde pupilar.
- **Watershed:** permitió obtener una máscara compacta de la pupila mediante marcadores internos y externos. Sin embargo, el contorno obtenido no coincidió completamente con el borde observado en la imagen.

**Conclusión:** K-Means y Watershed presentaron los resultados más prometedores. Se propone repetir la segmentación sobre la ROI procesada con CLAHE para evaluar si la mejora del contraste local favorece la delimitación de la pupila.

## Módulo 3. Extracción de características y dinámica temporal

**Observaciones:** Pendiente.

## Módulo 4. Compresión de imágenes

**Observaciones:** Pendiente.
