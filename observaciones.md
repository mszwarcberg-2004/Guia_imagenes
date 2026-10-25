# Observaciones y resultados — TP Imágenes Biomédicas

Video de trabajo: `video5.mov` (59,97 FPS, 12,14 s). La ROI (241 × 161 px) y el contorno de referencia a mano se hicieron sobre un frame de este video.

---

## Módulo 1. Procesamiento de imágenes

### 1.1 Carga, muestreo y espacios de color

**Frame y ROI.** Tomamos un frame del video y recortamos una ROI alrededor de la pupila y parte del iris. En el frame completo se ve bien el ojo y la ROI muestra el contorno de la pupila con más detalle. Cerca del borde de la pupila hay un reflejo del flash que puede complicar la detección de bordes y la segmentación.

**Dimensiones y bits.** La ROI mide 241 × 161 px, con 3 canales (RGB) y 8 bits por canal, es decir 24 bits por píxel. La pupila ocupa unos 70 px de diámetro (ver 3.1.1), suficiente para delimitar su borde.

**Submuestreo.**

| Resolución | Tamaño | Qué se ve |
|---|---|---|
| 100 % | 241 × 161 | Pupila e iris distinguibles, algo pixelada |
| 50 % | 120 × 80 | Pupila reconocible, bordes menos definidos |
| 25 % | 60 × 40 | Contorno irregular, casi sin textura |
| 5 % | 12 × 8 | No se distingue nada |

Al bajar la resolución se pierde primero el contorno y después la forma de la pupila, así que con poca resolución las mediciones de área y diámetro serían mucho menos precisas.

**Cuantización.** Con 4 bits las transiciones se vuelven más bruscas, pero la pupila se reconoce bien. Con 2 bits quedan zonas planas y casi sin textura, aunque la forma general se mantiene. Cuantizar reduce los niveles de gris, no la cantidad de píxeles.

**Espacios de color.** En RGB la pupila es oscura en los tres canales, las diferencias entre canales son chicas y el reflejo aparece en todos. En HSV, H es ruidoso donde hay poca saturación, S muestra variaciones entre regiones y V es la que mejor separa la pupila del iris. En V el reflejo sigue muy brillante, así que cambiar de espacio de color no lo elimina.

En resumen, hay que conservar resolución y profundidad de bits suficientes para no perder el contorno de la pupila, y los reflejos del flash van a seguir siendo un problema en las etapas siguientes.

### 1.2 Calidad, transformaciones de intensidad y filtrado espacial

**Brillo y contraste.** Sumar 50 desplaza el histograma hacia la derecha y multiplicar por 1,5 lo ensancha. En los dos casos muchos píxeles llegan a 255 y se saturan, y esa información no se recupera.

**Escalado por rango de interés.** Usamos los percentiles 5 y 95 de la ROI (f1 = 8 y f2 = 129) y expandimos ese intervalo a 0–255. Se separa mejor la pupila del iris, pero se saturan algunas zonas del iris y se pierde textura.

**Ecualización.** El escalado lineal satura algunas regiones. La ecualización global mejora mucho el contraste, pero deja zonas muy claras y exagera algunas texturas. CLAHE mejora menos, pero conserva mejor el detalle del iris: es el resultado más equilibrado para esta ROI. No vemos ruido marcado a simple vista, aunque la imagen tiene reflejos.

**Ruido y filtros.** Probamos media, mediana y gaussiano (3 × 3) y simulamos ruido gaussiano, sal y pimienta y speckle. La media suaviza pero desdibuja el borde de la pupila, la mediana es la más útil contra sal y pimienta, y el gaussiano reduce variaciones suaves a costa de algo de detalle fino. Ningún filtro elimina el reflejo del flash.

### 1.3 Bordes y dominio frecuencial

**Detección de bordes.** Aplicamos los cinco operadores a la ROI original y a la filtrada con gaussiano.

| Operador | Resultado |
|---|---|
| Roberts | Detecta el contorno, pero también el reflejo y variaciones del iris |
| Sobel y Prewitt | Parecidos entre sí, bordes gruesos |
| Laplaciano | Contorno poco definido y muchos detalles extra |
| Canny | Contorno fino y casi cerrado; también marca el reflejo |

Canny es el mejor para identificar el borde. El filtro gaussiano previo no mejora la continuidad del contorno.

**Fourier 2D.** La magnitud se concentra en el centro del espectro, o sea en las bajas frecuencias. Si reconstruimos solo con la magnitud, la imagen deja de ser reconocible; con solo la fase se conservan los contornos. Al cruzar magnitud y fase de la imagen original y la filtrada, el resultado se parece a la imagen de la que viene la fase, y como las dos son muy parecidas los cruces quedan casi iguales a la ROI original. Es la fase la que guarda la información estructural.

**PSD.** La potencia está concentrada en las bajas frecuencias y el filtro gaussiano reduce las altas. Pero el borde de la pupila también es de alta frecuencia, así que filtrar más no garantiza detectar mejor la pupila.

**Filtrado en frecuencia.** Usamos filtros ideales con corte en el 10 % del lado menor del frame. El pasa-bajos suaviza todo, pierde pestañas y textura del iris y produce *ringing* alrededor de los bordes fuertes, típico del filtro ideal. El pasa-altos resalta bordes (ojo, pestañas, pupila). El gaussiano espacial suaviza menos y conserva mejor la definición, así que con estos parámetros es preferible al pasa-bajos ideal.

**PSNR y SSIM** (cada filtro contra la ROI original):

| Filtro | PSNR (dB) | SSIM |
|---|---:|---:|
| Media | 39,04 | 0,9779 |
| Mediana | 41,49 | 0,9851 |
| Gaussiano | 41,27 | 0,9868 |

La mediana tiene el mayor PSNR y el gaussiano el mayor SSIM; los tres quedan muy cerca de la original. Estas métricas miden parecido con la imagen original, no cuánto ruido o reflejo se eliminó.

Respuestas a las preguntas de la guía: ningún filtro elimina el reflejo y Canny sobre la ROI original da el mejor contorno; la calidad frente a la imagen original se cuantifica con PSNR (diferencia de intensidad) y SSIM (similitud estructural).

---

## Módulo 2. Segmentación y morfología

### 2.1 Segmentación de la pupila

Probamos Otsu, K-Means (K de 2 a 5), Snakes y Watershed sobre la ROI en grises y comparamos cada máscara con el contorno marcado a mano usando el coeficiente Dice.

| Método | Dice |
|---|---:|
| Otsu | 0,7849 |
| K-Means K=2 | 0,7972 |
| K-Means K=3 | 0,8893 |
| K-Means K=4 | 0,9485 |
| K-Means K=5 | 0,9498 |
| Snakes | **0,9764** |
| Watershed | 0,9659 |

Snakes fue el mejor, seguido por Watershed. Otsu y K-Means con K = 2 incluyen píxeles que no son pupila y quedan cerca de 0,79; K-Means mejora al subir K y llega a 0,95.

### 2.2 Morfología

Sobre la máscara de Snakes aplicamos erosión, dilatación, apertura y cierre y comparamos con la referencia.

| Operación | Dice | IoU |
|---|---:|---:|
| Sin morfología | **0,9764** | **0,9539** |
| Erosión | 0,9728 | 0,9471 |
| Dilatación | 0,9632 | 0,9290 |
| Apertura | 0,9764 | 0,9539 |
| Cierre | 0,9763 | 0,9536 |

Sin morfología y apertura empatan, y el cierre da prácticamente lo mismo. La erosión achica la máscara y la dilatación la agranda, y las dos empeoran el resultado. Nos quedamos con Snakes sin morfología, que es lo más simple. La esqueletización (2.2.2) da una versión adelgazada de la máscara que sirve para describir la forma, pero no se usó para calcular el área.

Esta elección se hizo con un solo frame de referencia, así que vale para esa imagen y no asegura que sea el mejor método para todos los frames ni para otros videos.

---

## Módulo 3. Características y dinámica temporal

### 3.1 Descriptores del frame segmentado

**Morfometría** (máscara final, Snakes sin morfología):

| Descriptor | Valor |
|---|---:|
| Área | 3847 px² |
| Perímetro | 232,45 px |
| Centroide | (116,75; 76,12) px |
| Circularidad | 0,8947 |
| Excentricidad | 0,3575 |
| Diámetro equivalente | 69,99 px |

La circularidad cercana a 1 y la excentricidad baja indican una región casi circular. El diámetro sale de `D = 2·√(A/π)` y está en píxeles, sin calibrar a milímetros.

**Textura de primer orden (pupila).**

| Canal | Media | Desvío | Entropía | Asimetría |
|---|---:|---:|---:|---:|
| R | 11,90 | 9,67 | 3,74 | 3,39 |
| G | 9,04 | 8,05 | 3,53 | 3,49 |
| B | 11,43 | 7,03 | 4,04 | 2,35 |

La asimetría positiva alta indica que hay algunos píxeles mucho más brillantes que el resto, probablemente el reflejo. Estos números describen las intensidades de la región, no la textura anatómica.

**GLCM de la pupila** (distancia de 1 px):

| Ángulo | Contraste | Energía | Homogeneidad | Correlación | Entropía |
|---|---:|---:|---:|---:|---:|
| 0° | 0,2650 | 0,3074 | 0,9002 | 0,8796 | 2,3499 |
| 45° | 0,4555 | 0,2814 | 0,8612 | 0,7703 | 2,4799 |
| 90° | 0,2560 | 0,3054 | 0,8984 | 0,8821 | 2,3534 |
| 135° | 0,3594 | 0,2792 | 0,8711 | 0,8310 | 2,4767 |

Los valores cambian con la orientación: en 45° y 135° el contraste es mayor que en 0° y 90°, así que la relación entre píxeles vecinos no es igual en todas las direcciones.

**Textura del iris por canal.** No segmentamos el iris: usamos como aproximación un anillo alrededor de la pupila, entre 1,15 y 1,65 radios.

| Canal | Media | Desvío | Entropía | Asimetría |
|---|---:|---:|---:|---:|
| R | 102,67 | 12,38 | 5,37 | 0,68 |
| G | 87,36 | 11,32 | 5,22 | 0,84 |
| B | 61,23 | 10,37 | 5,08 | 1,05 |

| Canal | Contraste | Energía | Homogeneidad | Correlación | Entropía |
|---|---:|---:|---:|---:|---:|
| R | 0,4396 | 0,1047 | 0,8184 | 0,9075 | 3,9453 |
| G | 0,4175 | 0,1248 | 0,8245 | 0,8921 | 3,7447 |
| B | 0,4439 | 0,1314 | 0,8189 | 0,8746 | 3,6943 |

El canal R tiene el mayor desvío y la mayor entropía (en primer y segundo orden), la menor energía y la menor homogeneidad, así que es el que aporta más variedad de textura. B tiene el contraste más alto, pero por muy poco (0,4439 contra 0,4396). La conclusión depende del anillo elegido y de un único frame.

### 3.2 Seguimiento temporal y respuesta fisiológica

#### Procesamiento del video

Primero se estima automáticamente una "semilla": la posición y el radio típicos de la pupila, a partir de 15 frames repartidos en el video. Dio centro en (412,6; 345,7) px y radio de 39,2 px, con los 15 frames consistentes. Después, frame a frame, se busca la pupila cerca de la última posición y se segmenta con el método del Módulo 2 (Snakes, sin morfología), controlando que el área, la forma y los saltos entre frames sean razonables. Este seguimiento no usa la ROI marcada a mano.

Se procesaron 728 frames. El 94,09 % tuvo detección válida y 43 quedaron sin detección. La mayoría de los intentos descartados fueron por área fuera de rango (43) o por saltos de área entre frames (17).

*Validación manual.* Marcamos el contorno a mano en algunos frames del registro y comparamos área y diámetro con las mediciones automáticas.

| Error | Área | Diámetro |
|---|---:|---:|
| MAE | — | — |
| RMSE | — | — |


#### Curva temporal

El área baja después de que se enciende la luz y se recupera en parte cuando se apaga. En la curva también aparecen huecos y picos por parpadeos, por el párpado que tapa la pupila y por los cambios de iluminación; esos tramos no son cambios fisiológicos.

#### Parámetros fisiológicos

El registro es de tipo A: la luz se enciende a los 1,98 s y se apaga a los 9,1 s. La constricción se mide desde el encendido y la redilatación desde el apagado.

| Parámetro | Valor |
|---|---:|
| Encendido de la luz | 1,984 s |
| Apagado de la luz | 9,105 s |
| Área basal | 8545 px² |
| Área mínima | 4006 px² (a los 5,50 s) |
| Amplitud | 4539 px² (53,1 % del basal) |
| Inicio de la constricción | 2,135 s |
| Latencia | 0,150 s |
| Velocidad de constricción | 1347,48 px²/s |
| Velocidad de redilatación | 875,01 px²/s |

Cómo se calcula cada uno:
- **Área basal:** mediana del área en los 0,5 s anteriores al encendido.
- **Amplitud:** área basal menos área mínima.
- **Latencia:** tiempo entre el encendido y el inicio de la constricción, que se marca cuando el área queda 0,1 s seguidos por debajo del basal menos el 5 % de la amplitud.
- **Velocidad de constricción:** amplitud dividida por el tiempo entre el inicio y el mínimo.
- **Velocidad de redilatación:** cambio de área entre el apagado y el final de la ventana, dividido por ese tiempo. La ventana es de hasta 4 s; acá son unos 3 s porque el video termina a los 12,14 s.

Los valores tienen sentido: la redilatación es más lenta que la constricción, como se espera, y la latencia de 0,15 s es razonable aunque algo menor que lo habitual (0,2 a 0,4 s), seguramente porque el inicio se marca con una caída de solo el 5 % de la amplitud.

#### Cambios respecto de la primera versión

La primera versión del código dio resultados que no podían ser correctos, y los corregimos:

| Parámetro | Primera versión | Versión final |
|---|---:|---:|
| Encendido | 1,934 s | 1,984 s |
| Inicio de la constricción | 1,951 s | 2,135 s |
| Área basal | 8454 px² | 8545 px² |
| Área mínima | 3532 px² (a los 4,19 s) | 4006 px² (a los 5,50 s) |
| Amplitud | 4922 px² | 4539 px² |
| Latencia | 0,017 s | 0,150 s |
| Vel. de constricción | 2202,67 px²/s | 1347,48 px²/s |
| Vel. de redilatación | 378,57 px²/s | 875,01 px²/s |

- **Latencia de 0,017 s.** Equivale a un solo frame (1/59,97 s), o sea que el inicio se marcaba justo después del estímulo. Ahora el inicio exige que la caída se sostenga 0,1 s y se mide contra un área basal calculada solo con los 0,5 s previos, y si la amplitud es comparable al ruido no se informa latencia.
- **Área mínima.** Antes se tomaba el valor más bajo de toda la serie, que podía ser un solo frame mal segmentado. Ahora se busca el mínimo sobre una mediana móvil de 0,25 s, y el mínimo pasó de 3532 a 4006 px²: el valor anterior era un hundimiento aislado y por eso la amplitud y la velocidad de constricción estaban infladas.
- **Instante del encendido.** Antes se tomaba el punto de mayor cambio de luminosidad, que cae a mitad de la subida. Ahora es el primer cruce del 20 % del escalón de luz, que se parece más al momento real del estímulo. El mismo método detecta el apagado, y según cuál de los dos aparece primero clasifica el registro como A (luz que se enciende) o B (luz que se apaga).
- **Redilatación.** Antes se medía desde el mínimo hasta la última muestra, mezclando el tiempo con la luz encendida. Ahora se mide desde el apagado.


#### Variabilidad entre registros

Los parámetros se calcularon para los cinco videos de la misma persona.

> **Pendiente:** completar con la tabla que da `comparar_registros.py` (o con `resultados/variabilidad_entre_registros.csv`) usando la versión final del código. Las tablas anteriores no sirven porque tenían latencias de 0 s y de más de 2 s, que eran errores del método viejo.

Para comparar entre videos conviene usar la amplitud relativa (% del basal) y no los px², porque el ojo puede ocupar un tamaño distinto en el encuadre de cada video.

#### Artefactos

Las fuentes principales de artefactos son los parpadeos (en este video, cerca de 1,8 s y entre 9,2 y 9,6 s), el movimiento del párpado que tapa parte de la pupila, los cambios bruscos de iluminación y las detecciones incorrectas que producen picos de área.

Para tratarlos, el código:
1. descarta los frames donde la detección falla o el área o la posición saltan demasiado respecto del frame anterior;
2. elimina valores atípicos con el filtro de Hampel y con una mediana local de 0,3 s;
3. descarta las mediciones pegadas a cada parpadeo (0,05 s a cada lado) y los tramos válidos de menos de 0,1 s;
4. interpola solo huecos de hasta 3 frames y aplica una mediana de 5 frames;
5. deja el resto como `NaN` y marca en una columna cada parpadeo, pérdida de seguimiento y salto del ojo.

> **Pendiente:** completar cuántos parpadeos y mediciones descartadas informó la terminal en la corrida final.

Aun así pueden quedar picos sueltos, así que el control de artefactos es parcial.

**Conclusión del Módulo 3.** Logramos seguir la pupila en todo el video, obtener la curva de área y calcular los parámetros del reflejo fotomotor. La parte más delicada es la robustez frente a parpadeos y cambios de luz.

---

## Módulo 4. Compresión

### 4.1 Sin pérdida: Huffman y RLE

Aplicamos Huffman y RLE (horizontal y vertical) a la ROI en grises, la máscara binaria y el frame completo. El ratio es el tamaño original dividido por el tamaño codificado estimado.

| Imagen | Huffman | RLE horizontal | RLE vertical |
|---|---:|---:|---:|
| ROI (161 × 241) | 1,196 | 0,398 | 0,398 |
| Máscara pupilar (161 × 241) | 8,000 | 91,728 | 86,803 |
| Frame completo (598 × 968) | 1,086 | 0,464 | 0,456 |

En la máscara binaria, con grandes zonas constantes, los dos métodos comprimen mucho y RLE muchísimo más que Huffman (casi 92 contra 8). En la ROI y el frame, donde las intensidades cambian todo el tiempo, RLE da un ratio menor a 1: lo codificado ocupa más que el original.

Son ratios teóricos: Huffman cuenta solo la longitud de los códigos y RLE usa 8 bits por intensidad y 16 por cantidad de repeticiones, sin encabezados ni tablas. Al no tener pérdida, el PSNR sería infinito y el SSIM igual a 1, pero no verificamos la reconstrucción decodificando.

### 4.2 JPEG

Probamos Q = 90, 50, 10 y 5. Al bajar Q aumenta el ratio y bajan el PSNR y el SSIM.

**ROI**

| Q | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| 90 | 4,813 | 46,22 | 0,9919 |
| 50 | 10,660 | 39,66 | 0,9669 |
| 10 | 25,867 | 31,52 | 0,8283 |
| 5 | 36,229 | 28,12 | 0,6986 |

**Máscara pupilar**

| Q | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| 90 | 18,380 | 53,33 | 0,9978 |
| 50 | 26,395 | 38,95 | 0,9817 |
| 10 | 36,536 | 32,00 | 0,9740 |
| 5 | 39,392 | 27,40 | 0,1930 |

**Frame completo**

| Q | Ratio | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| 90 | 5,803 | 46,21 | 0,9923 |
| 50 | 13,767 | 39,15 | 0,9695 |
| 10 | 35,576 | 31,22 | 0,8569 |
| 5 | 51,670 | 27,63 | 0,7643 |

JPEG llega a ratios altos, sobre todo en el frame completo (51,7 con Q = 5), a cambio de perder calidad. En la máscara binaria el PSNR baja de a poco, pero el SSIM se mantiene alto hasta Q = 10 (0,97) y cae de golpe con Q = 5 (0,19): con una compresión tan fuerte se destruyen los bordes abruptos de la máscara.

### 4.3 Efectos visuales y medición

Con Q = 90 la imagen es casi igual a la original y con Q = 50 el contorno de la pupila se conserva bien. Los bloques y la pérdida de textura se hacen evidentes desde Q = 10, y más con Q = 5. La pérdida afecta sobre todo a los bordes, justo lo que se usa para medir la pupila; en las zonas uniformes se ven bloques y bandas, pero la degradación es menos notoria.

Para ver el efecto sobre el diámetro volvimos a segmentar la ROI original y las versiones JPEG con Watershed + dilatación:

| Imagen | Diámetro (px) | Diferencia |
|---|---:|---:|
| ROI original | 72,80 | — |
| JPEG Q=50 | 72,87 | +0,08 |
| JPEG Q=10 | 74,28 | +1,49 |

Con Q = 50 el diámetro casi no cambia; con Q = 10 se sobreestima en 1,49 px, cerca de un 2 %. Este resultado vale para esta imagen y este método. El diámetro original de la tabla (72,80 px) no coincide con el de 3.1 (69,99 px) porque acá se segmentó con Watershed + dilatación y en el Módulo 3 con Snakes sin morfología.

**Conclusión del Módulo 4.** Los métodos sin pérdida sirven cuando hay que conservar la información exacta, y RLE rinde muchísimo en la máscara binaria. JPEG comprime más pero degrada. Para medir la pupila conviene una calidad moderada (Q = 50 o más).

---

## Limitaciones

- El video analizado dura 12,14 s y el protocolo propone unos 30 s.
- El método de segmentación se eligió con un solo frame marcado a mano en `video5`; el seguimiento temporal es automático, pero ese método no se validó con otros videos.
- Los parpadeos y los cambios de luz siguen generando picos; el control es parcial.
- La latencia depende del criterio de inicio (caída del 5 % de la amplitud sostenida 0,1 s), por eso puede quedar algo por debajo de lo habitual.
- Áreas y diámetros están en píxeles, sin calibrar a milímetros.