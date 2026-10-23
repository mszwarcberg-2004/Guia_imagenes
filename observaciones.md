
# Observaciones y resultados — TP Imágenes Biomédicas

## Módulo 1. Procesamiento de imágenes

### 1.1 Carga, muestreo y espacios de color

#### 1.1.1 Extracción del frame y selección de la ROI

Se trabajó con un frame representativo del video seleccionado y una región de interés (ROI) centrada en la pupila y parte del iris, para estudiar las estructuras de interés.

**Aclaración:** En la ejecución actual se seleccionó `video5.mp4`, pero el programa cargó `imagenes/roi.png` porque ya existía una ROI guardada. Por eso, los análisis de ROI y máscara deben interpretarse como resultados de esas imágenes guardadas hasta verificar que efectivamente proceden del mismo video. El análisis temporal del Módulo 3 sí se realizó sobre `video5.mp4`.

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

### 1.2 Calidad, transformaciones de intensidad y filtrado espacial

#### 1.2.1 Histograma, brillo y contraste

Se calculó el histograma del frame completo en escala de grises y se aplicaron dos transformaciones: aumento de brillo mediante la suma de 50 unidades y modificación del contraste mediante la multiplicación de las intensidades por 1,5.

En ambos casos, el histograma se desplazó hacia intensidades mayores y se observa mucha saturación, debido a que numerosos píxeles alcanzaron el valor máximo de 255, provocando pérdida de información.

#### 1.2.2 Escalado lineal por rango de interés

Se seleccionaron automáticamente los límites del intervalo utilizando los percentiles 5 y 95 de las intensidades de la ROI en escala de grises, para que el intervalo seleccionado contenga aproximadamente el 90% central de los valores de intensidad, excluyendo los extremos de la distribución.

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

Se comparó la ROI original con versiones filtradas mediante media, mediana y gaussiano (ventanas 3 × 3). Además, se simularon ruido gaussiano, impulsivo (sal y pimienta) y speckle, y se aplicaron filtros espaciales para evaluar su atenuación.

**Observaciones:**

- **Filtro de media:** suaviza variaciones locales, pero también puede desdibujar el borde de la pupila.
- **Filtro de mediana:** resulta especialmente útil para atenuar el ruido impulsivo.
- **Filtro gaussiano:** reduce fluctuaciones suaves a costa de cierta pérdida de detalles finos.

En las imágenes con ruido simulado, los filtros atenúan parcialmente las alteraciones introducidas, pero ninguno elimina el reflejo luminoso intenso sin modificar otras estructuras.

No se cuenta con una imagen real libre de ruido para cuantificar la eliminación de ruido original. La evaluación se basa en comparaciones visuales y en las métricas PSNR/SSIM expuestas más adelante, que cuantifican similitud respecto de la ROI original, pero no garantizan una mejor segmentación.

### 1.3 Detección de bordes y dominio frecuencial

#### 1.3.1 Detección de bordes

Se aplicaron los operadores Roberts, Sobel, Prewitt, Laplaciano y Canny sobre la ROI original y sobre la ROI filtrada mediante un filtro gaussiano de 3 × 3, para comparar la detección y continuidad del contorno pupilar.

**Observaciones:**

- **Roberts:** permite identificar el contorno de la pupila, aunque también resalta el reflejo luminoso y algunas variaciones de intensidad del iris.
- **Sobel y Prewitt:** presentan resultados similares, detectando claramente el contorno pupilar, aunque con bordes relativamente gruesos.
- **Laplaciano:** resalta variaciones bruscas de intensidad, pero el contorno de la pupila aparece menos definido y se detectan detalles adicionales.
- **Canny:** produce el contorno más fino y prácticamente continuo, aunque también identifica bordes correspondientes a los reflejos luminosos.

Al comparar la imagen original con la filtrada, se observa que el filtro gaussiano suaviza algunas variaciones de intensidad y reduce detalles pequeños. Sin embargo, no produce una mejora significativa en la continuidad del contorno pupilar.

En esta ROI, **Canny presenta el resultado más conveniente para identificar el borde de la pupila**, ya que genera un contorno delgado y prácticamente cerrado.

#### 1.3.2 Transformada de Fourier 2D

Se calculó la Transformada de Fourier 2D de la ROI original y se representó su magnitud en escala logarítmica y su fase.

Luego, se reconstruyó la imagen mediante la transformada inversa utilizando únicamente la magnitud, únicamente la fase y combinando magnitudes y fases de la ROI original y la filtrada con gaussiano.

**Observaciones:**

- **Magnitud:** presenta una mayor concentración de valores en la región central del espectro, lo que indica un predominio de bajas frecuencias espaciales.
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

#### Evaluación cuantitativa mediante PSNR y SSIM

Se calcularon las métricas PSNR y SSIM para comparar la ROI original con las imágenes obtenidas mediante los filtros espaciales de media, mediana y gaussiano.

Se obtuvieron los siguientes resultados:

| Filtro | PSNR (dB) | SSIM |
|---|---:|---:|
| Media | 39,57 | 0,9769 |
| Mediana | 42,42 | 0,9834 |
| Gaussiano | 41,75 | 0,9858 |

**Observaciones:**

El filtro de mediana presentó el valor más alto de PSNR, lo que indica una menor diferencia de intensidad respecto de la ROI original. Por otro lado, el filtro gaussiano obtuvo el mayor SSIM, mostrando una mayor similitud estructural con la imagen original.

Los tres filtros presentaron valores elevados de ambas métricas, indicando que las imágenes procesadas mantienen una alta similitud con la ROI original.

Sin embargo, estas métricas no permiten determinar por sí solas qué filtro elimina mejor el ruido o los reflejos luminosos, ya que únicamente evalúan la similitud respecto de la imagen de referencia.

En esta comparación, **el filtro gaussiano presentó la mayor similitud estructural**, aunque no eliminó el reflejo luminoso presente sobre el borde de la pupila.

#### Conclusiones generales del apartado 1.3

A partir de las comparaciones realizadas, se observó que **Canny permite identificar de manera más clara y continua el contorno pupilar**, tanto en la ROI original como en la filtrada.

El filtro gaussiano permite suavizar pequeñas variaciones de intensidad, pero no elimina los reflejos luminosos del flash, que continúan generando bordes adicionales.

Por otra parte, el análisis de Fourier y de la PSD permitió comprobar que el filtrado modifica la distribución de las componentes frecuenciales y que la fase contiene información fundamental sobre la estructura espacial de la imagen.

En conjunto, los resultados muestran que **no siempre es conveniente aplicar un filtrado previo**, ya que la reducción de ruido o detalles también puede afectar los bordes que se desean conservar.

Finalmente, la evaluación mediante PSNR y SSIM mostró que los tres filtros espaciales conservan una alta similitud con la ROI original. El filtro de mediana obtuvo el mayor PSNR, mientras que el gaussiano presentó el mayor SSIM. Sin embargo, estas métricas deben complementarse con el análisis visual de los bordes.

---

## Módulo 2. Segmentación y morfología matemática

### 2.1 Segmentación de la pupila — ROI original

Se evaluaron Otsu, K-Means (K=2 a 5), Snakes y Watershed sobre la ROI en escala de grises. Se compararon sus máscaras con una referencia manual mediante el coeficiente Dice.

| Método | Dice |
|---|---:|
| Otsu | 0,8240 |
| K-Means K=2 | 0,8429 |
| K-Means K=3 | 0,9065 |
| K-Means K=4 | 0,9065 |
| K-Means K=5 | 0,9096 |
| Snakes | 0,9292 |
| Watershed | **0,9434** |

**Observaciones:**

Otsu permitió detectar regiones oscuras, pero incluyó píxeles fuera de la pupila. K-Means mejoró al aumentar la cantidad de grupos.

Snakes delimitó gran parte de la pupila y alcanzó un Dice de 0,9292, por lo que no corresponde afirmar que solo se contrajo hacia el reflejo.

Watershed obtuvo el mayor Dice y se seleccionó para el siguiente apartado.

Los reflejos luminosos y el tamaño reducido de la ROI siguen siendo limitaciones relevantes.

### 2.2 Operaciones morfológicas y selección de máscara final

Se aplicaron erosión, dilatación, apertura y cierre sobre la máscara obtenida con Watershed, y se compararon con la referencia manual.

| Operación | Dice | IoU |
|---|---:|---:|
| Sin morfología | 0,9434 | 0,8928 |
| Erosión | 0,8874 | 0,7976 |
| **Dilatación** | **0,9563** | **0,9163** |
| Apertura | 0,9425 | 0,8913 |
| Cierre | 0,9434 | 0,8928 |

La **dilatación** dio el mejor resultado, por lo que se seleccionó **Watershed + dilatación** como combinación final.

La erosión redujo demasiado la región segmentada.

La esqueletización permitió representar una versión adelgazada de la máscara, útil como descripción estructural, aunque no reemplaza el contorno ni se utilizó para calcular el área.

**Conclusión del Módulo 2:**

La combinación Watershed + dilatación obtuvo Dice = 0,9563 e IoU = 0,9163 respecto de la referencia manual de la ROI guardada.

Esta selección es válida para esa imagen; no demuestra que el mismo método sea el óptimo para todos los frames o para videos nuevos.

---

## Módulo 3. Extracción de características y dinámica temporal

### 3.1 Descriptores morfométricos y de textura

#### 3.1.1 Morfometría

Sobre la máscara final de la ROI guardada, obtenida con Watershed + dilatación, se calcularon las siguientes características:

| Descriptor | Resultado |
|---|---:|
| Área | 702 px² |
| Perímetro | 99,50 px |
| Centroide | (42,27; 24,57) px |
| Circularidad | 0,8911 |
| Excentricidad | 0,3026 |
| Diámetro equivalente | 29,90 px |

La circularidad relativamente próxima a 1 y la excentricidad baja indican que la región segmentada es aproximadamente circular.

El diámetro equivalente se obtuvo a partir del área mediante `D = 2 √(A/π)`; no representa una medición calibrada en milímetros.

#### 3.1.2 Textura de primer orden

Se calcularon media, desvío estándar, entropía y asimetría por canal RGB.

En la pupila se obtuvieron los siguientes valores:

| Canal | Media | Desvío estándar | Entropía | Asimetría |
|---|---:|---:|---:|---:|
| R | 46,4259 | 15,0527 | 4,8748 | 3,8201 |
| G | 36,9330 | 14,0934 | 4,7258 | 4,9158 |
| B | 39,1282 | 12,8720 | 4,8620 | 5,1439 |

Las asimetrías positivas elevadas son compatibles con la presencia de algunos píxeles más brillantes, posiblemente relacionados con reflejos.

Estos resultados describen la distribución de intensidades de la región estudiada, no necesariamente la textura anatómica pura.

#### 3.1.3 Textura de segundo orden (GLCM)

En la pupila se calcularon contraste, energía, homogeneidad, correlación y entropía para distancias de 1 píxel y ángulos de 0°, 45°, 90° y 135°.

Las métricas variaron según la orientación, lo que muestra que las relaciones locales de intensidad no son idénticas en todas las direcciones.

#### 3.1.4 Análisis de textura del iris

Se compararon los canales RGB usando las métricas GLCM:

| Canal | Contraste | Energía | Homogeneidad | Correlación | Entropía |
|---|---:|---:|---:|---:|---:|
| R | 0,4099 | 0,1234 | 0,8224 | 0,8646 | 3,5804 |
| G | 0,3799 | 0,1411 | 0,8412 | 0,8596 | 3,3992 |
| B | 0,4108 | 0,1374 | 0,8265 | 0,8577 | 3,4993 |

El canal **R** presentó la mayor entropía y una energía menor, lo que sugiere una distribución más variada de las combinaciones de intensidades.

El canal **B** presentó un contraste apenas mayor.

Por lo tanto, **R parece aportar mayor riqueza textural según la entropía**, aunque esa conclusión depende de las métricas, la cuantización y la región de iris elegida.

No se puede identificar un canal universalmente mejor a partir de un único frame.

### 3.2 Seguimiento temporal y respuesta fisiológica

#### 3.2.1 Procesamiento del video

Se analizaron 728 frames de `video5.mp4`, a 59,97 FPS, con una duración aproximada de 12,14 segundos.

Se utilizó Watershed + dilatación para seguir la pupila, con un **96,70% de detecciones válidas** según los criterios automáticos.

- Frames sin detección: 24.
- Valores atípicos descartados mediante Hampel: 9.
- Frames totales procesados: 728.

**Validación manual mediante MAE y RMSE:**

Se compararon las mediciones automáticas con contornos dibujados manualmente en 6 frames distribuidos a lo largo del registro.

| Error | Resultado |
|---|---:|
| MAE del área | 341,17 px² |
| RMSE del área | 375,72 px² |
| MAE del diámetro | 3,21 px |
| RMSE del diámetro | 3,54 px |

Los errores muestran diferencias apreciables entre ambas mediciones.

En varios de los frames seleccionados, el algoritmo estimó un diámetro superior al manual, aunque la evaluación debe considerar también la variabilidad del trazado manual y el tamaño reducido de la muestra.

Los seis frames permiten una primera validación, pero no aseguran precisión uniforme en el video completo.

#### 3.2.2 Curva temporal

Se generó la evolución del área pupilar en función del tiempo.

La curva muestra una disminución posterior al aumento de iluminación, seguida de una recuperación parcial.

Sin embargo, se identificaron picos abruptos y huecos asociados con parpadeos, oclusión parcial y cambios de iluminación.

Por lo tanto, no todos los valores automáticos representan cambios fisiológicos reales.

#### 3.2.3 Parámetros fisiológicos

El algoritmo informó los siguientes resultados para `video5.mp4`:

| Parámetro | Estimación automática |
|---|---:|
| Encendido del estímulo | 1,934 s |
| Apagado del estímulo | 9,105 s |
| Área basal | 5862 px² |
| Área mínima | 2577 px² |
| Amplitud | 3285 px² |
| Latencia | 0,083 s |
| Velocidad de constricción | 887,35 px²/s |
| Velocidad de redilatación | 298,12 px²/s |

Estas cifras son **estimaciones automáticas preliminares**.

La latencia muy corta y los errores de área durante parpadeos requieren revisión antes de considerarlas medidas fisiológicas confiables.

#### Variabilidad entre registros

Se procesaron automáticamente cinco videos de la misma persona mediante un script independiente.

Se obtuvieron parámetros para cuatro de ellos; en `video3.mov` no se identificó el estímulo.

| Registro | Latencia (s) | Amplitud (px²) | Amplitud relativa (%) |
|---|---:|---:|---:|
| video1.mov | 2,123 | 42,50 | 5,73 |
| video2.mov | 0,000 | 422,67 | 52,18 |
| video3.mov | No detectado | — | — |
| video4.mov | 0,501 | 198,00 | 16,23 |
| video5.mp4 | 0,083 | 3285,00 | 56,04 |

Entre los cuatro registros con parámetros calculados:

| Parámetro | Media | Desvío estándar |
|---|---:|---:|
| Latencia (s) | 0,677 | 0,989 |
| Amplitud relativa (%) | 32,545 | 25,316 |

La variabilidad observada puede reflejar tanto diferencias entre registros como errores del algoritmo.

En particular, las latencias de 0 y 2,123 s, así como las grandes diferencias de amplitud, hacen que **estos promedios no deban interpretarse como variabilidad fisiológica validada**.

La amplitud en píxeles² tampoco es directamente comparable si el tamaño aparente del ojo difiere entre videos, por lo que se reportó adicionalmente la amplitud relativa.

#### 3.2.4 Artefactos y tratamiento

Los principales artefactos observados fueron:

- Parpadeos y oclusiones pupilares.
- Movimientos del párpado que dificultan la segmentación.
- Cambios bruscos de iluminación.
- Picos de área asociados a detecciones incorrectas.

Se identificaron errores especialmente alrededor de 1,8 s, 4,1–4,2 s, 7,0–7,2 s y 9,1–9,5 s.

Para tratarlos, el procesamiento implementó:

- Restricciones sobre saltos de área y posición.
- Descarte de valores atípicos mediante el filtro Hampel.
- Valores faltantes representados como `NaN`.
- Interpolación de huecos de hasta tres frames.
- Suavizado mediante mediana temporal de cinco frames.

Pese a estas medidas, se conservaron algunos picos espurios, por lo que el control de artefactos es parcial.

#### Conclusión del Módulo 3

Se implementó la extracción de características del frame y el seguimiento pupilar temporal, incluyendo curvas, estimaciones fisiológicas, validación manual mediante MAE/RMSE y análisis exploratorio de variabilidad entre registros.

La principal limitación es la robustez del seguimiento ante parpadeos y cambios de iluminación.

Además, debe verificarse que las imágenes guardadas utilizadas en los análisis estáticos correspondan al mismo registro que el video analizado temporalmente.

---

## Módulo 4. Compresión de imágenes

### 4.1 Compresión sin pérdida: Huffman y RLE

Se analizaron la ROI en escala de grises, la máscara pupilar binaria y el frame completo mediante Huffman y RLE horizontal/vertical.

Se obtuvieron los siguientes **ratios teóricos**, definidos como tamaño original en memoria dividido por tamaño codificado estimado:

| Imagen | Huffman | RLE horizontal | RLE vertical |
|---|---:|---:|---:|
| ROI (56 × 85 px) | 1,344 | 0,400 | 0,402 |
| Máscara pupilar (56 × 85 px) | 8,000 | 26,011 | 24,410 |
| Frame completo (480 × 768 px) | 1,105 | 0,579 | 0,566 |

**Observaciones:**

Huffman presentó una compresión moderada en la ROI y el frame completo, pero una reducción importante en la máscara binaria.

RLE resultó especialmente eficiente para la máscara debido a sus grandes regiones de intensidad constante.

En la ROI y el frame completo, donde las intensidades cambian continuamente, RLE produjo ratios inferiores a 1, es decir, el tamaño codificado estimado fue mayor que el original.

Estos ratios **no representan archivos comprimidos completos**: Huffman contempla únicamente longitudes de códigos, y RLE considera 8 bits por intensidad más 16 bits por cantidad de repeticiones.

No se incorporaron encabezados ni tablas de decodificación.

Como Huffman y RLE son métodos sin pérdida, una reconstrucción exacta tendría SSIM = 1 y PSNR infinito, pero la implementación actual no verifica la reconstrucción mediante decodificación.

### 4.2 JPEG con pérdida y comparación cuantitativa

Se aplicó JPEG con calidades Q = 90, 50, 10 y 5.

A medida que disminuyó Q, aumentó el ratio de compresión y disminuyeron PSNR y SSIM.

#### Resultados de la ROI

| Calidad | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Q=90 | 3,987 | 46,23 | 0,9894 |
| Q=50 | 7,031 | 39,57 | 0,9570 |
| Q=10 | 10,721 | 32,38 | 0,8093 |
| Q=5 | 11,667 | 28,85 | 0,6732 |

#### Resultados de la máscara pupilar

| Calidad | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Q=90 | 4,718 | 46,82 | 0,9852 |
| Q=50 | 6,889 | 33,01 | 0,9117 |
| Q=10 | 9,616 | 25,94 | 0,8619 |
| Q=5 | 10,554 | 23,44 | 0,2794 |

La degradación afecta especialmente la preservación de una máscara binaria con bordes abruptos.

#### Resultados del frame completo

| Calidad | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Q=90 | 6,801 | 45,99 | 0,9921 |
| Q=50 | 16,278 | 39,50 | 0,9708 |
| Q=10 | 40,783 | 32,02 | 0,8610 |
| Q=5 | 57,717 | 28,43 | 0,7689 |

Los resultados muestran que JPEG permite alcanzar ratios de compresión elevados, especialmente sobre el frame completo, aunque la calidad disminuye progresivamente.

### 4.3 Efectos visuales y medición pupilar

#### Artefactos de bloque

En las comparaciones visuales, Q=90 mantuvo una apariencia muy similar a la original y Q=50 conservó razonablemente el contorno pupilar.

Los artefactos de bloque y la pérdida de textura se hicieron evidentes alrededor de **Q=10**, y fueron más pronunciados con Q=5.

#### Bordes frente a regiones uniformes

La pérdida de detalles afecta especialmente las transiciones de intensidad y los bordes de la pupila, importantes para estimar su tamaño.

En las regiones uniformes, la degradación puede ser menos perceptible, aunque aparecen bloques y bandas de intensidad.

En la máscara pupilar, las irregularidades de los bordes se observan claramente a bajas calidades.

#### Impacto sobre el diámetro pupilar

Se volvió a segmentar la ROI original y sus versiones reconstruidas con JPEG Q=50 y Q=10, empleando Watershed + dilatación.

Se calculó el diámetro equivalente en cada caso:

| Imagen | Diámetro (px) | Diferencia respecto del original |
|---|---:|---:|
| ROI original | 29,90 | — |
| JPEG Q=50 | 30,11 | +0,21 px |
| JPEG Q=10 | 31,23 | +1,33 px |

En esta ROI, JPEG Q=50 modificó poco la estimación del diámetro, mientras que Q=10 produjo una sobreestimación mayor.

Este resultado es específico de la imagen y el método empleados y no demuestra que el mismo efecto se repita en todos los frames o videos.

#### Conclusión del Módulo 4

Los métodos sin pérdida son especialmente adecuados cuando se necesita preservar exactamente la información, y RLE mostró ventajas notorias en la máscara binaria.

JPEG logró ratios altos sobre imágenes naturales, a costa de una degradación progresiva.

Para conservar la precisión de mediciones pupilares, los resultados favorecen calidades moderadas frente a una compresión muy agresiva.

---

## Consideraciones para la entrega y reproducibilidad

El programa principal ejecuta secuencialmente los Módulos 1 a 4.

Sin embargo, en el estado probado, `main.py` **reutiliza `imagenes/roi.png` y `imagenes/mascara_referencia.png` cuando existen**, aunque se cambie `VIDEO_PATH`.

Por lo tanto, al analizar un video nuevo no necesariamente se regeneran la ROI ni la referencia manual de ese registro.

Esto debe resolverse o documentarse como limitación antes de afirmar que el pipeline es completamente automático y reproducible sobre videos no vistos.

Asimismo, el registro analizado temporalmente (`video5.mp4`) dura 12,14 segundos, mientras que el protocolo de adquisición propone registros de aproximadamente 30 segundos.

Los resultados presentados corresponden a los archivos disponibles y no constituyen una validación exhaustiva del protocolo de adquisición.

