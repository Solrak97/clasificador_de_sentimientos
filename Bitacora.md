# Bitacora: detección de sentimientos utilizando redes neuronales
### Investigadores:
* Luis Carlos Quesada
* Mario Viquez
* Gianfranco Bagnarello
* Isaac Herrera
* Daniel Ramirez
---
 
### Indice:

* [Replicación de la topología base](#replicación-de-la-topología-base-issa-et-al.)
* [Busqueda de los datos de entrada al modelo propuesto](#busqueda-de-los-datos-de-entrada-al-modelo-propuesto)
* [Pruebas del modelo base](#pruebas-del-modelo-base)
* [Reconstrucción del modelo utilizando Pytorch en lugar de Keras](reconstrucción-del-modelo-utilizando-pytorch-en-lugar-de-keras)

---

## Replicación de la topología base Issa et al.
### Luis Carlos Quesada - 5/10/2022
Repliqué el modelo de Issa et al. El modelo está descrito en el paper pero describe la capa de entrada como 193 neuronas, anteriormente habíamos hecho extracción de caracteristicas pero el resultado era de más de 228 variables a diferencia de este, por lo que no podemos comprobar de momento que la red neuronal se ajuste a las predicciones descritas en el paper.

### Resultados:
Se obtuvo un modelo con la topología descrita en el paper a replicar, sin embargo no se puede comparar los resultados ya que es necesario investigar la dimensionalidad de los datos de entrada pues no fueron descritos en el modelo original. 

---
<br>
<br>

## Busqueda de los datos de entrada al modelo propuesto
### Luis Carlos Quesada - 5/18/2022
Para poder hacer una clasificación de audio usando la topología de red propuesta en el paper es necesario saber las dimensiones de entrada de los datos, la red tiene 193 variables de entrada y sabemos que hay algunass caracteristicas con tamaños fijos.

Se sabe que el tonnetz es de shape (n, 6) entonces al aplicar el calculo de medias el shape final es de (6,).

Chromagram produce 12 bins default, pero el tamaño de bins puede ser cualquier numero > 0.

El MFCC usualmente tiene entre 20-40ms por cada bin, sabemos que tenemos 3 segundos por audio, entonces podemos mover eta variabe en el rango [75, 120].

Mel Spectrogram de la misma forma depende de la cantidad de mels, que de nuevo es [75, 120], el tamaño de MFCC y spectrogram deben ser iguales.

Contrast se calcula con 6 bandas default pero funciona con n_bands > 1 y retorna n_bands + 1

Así que al final, tenemos algo así

<!-- $$ 
6 + (bands + 1) + bins + 2mel = 193
$$ --> 

<div align="center"><img style="background: white;" src="svg\nBcoiMI68J.svg"></div> 

<!-- $$
mel ∈ [75, 120]
$$ --> 

<div align="center"><img style="background: white;" src="svg\qeqZGbzVgF.svg"></div>
<!-- $$
bands > 0
$$ --> 

<div align="center"><img style="background: white;" src="svg\6FeOJvlnpQ.svg"></div>
<!-- $$
bins > 0
$$ --> 

<div align="center"><img style="background: white;" src="svg\OPBItwIF41.svg"></div>

### Resultados:
Por ahora asumiendo los valores default para n_bands y bins podemos movernos por el numero de mels, esto nos deja con 84 mels que es un valor valido ya que 
<!-- $$ 
75 <= mel <= 120 
$$ --> 

<div align="center"><img style="background: white;" src="svg\zL6jhbHNkE.svg"></div>

---
<br>
<br>

## Pruebas del modelo base
### Luis Carlos Quesada - 5/19/2022
Probé el modelo base con los datos de prueba a ver que pasa, creo que no estan para nada cerca pero podría ser por la forma en que los datos están codificados, estoy casi seguro de eso!
El modelo original fue entrenado usando softmax como salida para 8 variables, esto hace que sea necesaria una codificación un toque diferente a la que tengo actualmente, más cuando el optimizador es RMSprop y una perdida de sparse_entropy_loss, así que casi de fijo es eso

### Resultados
Las predicciones son como del 20% en promedio, es terrible pero creo que se puede arreglas mañana porque haciendo un max de las diferentes salidas, si se activa la salida "correcta" digamos, ahí se revisa un toque en la tarde.

---
<br>
<br>

## Reconstrucción del modelo utilizando Pytorch en lugar de Keras
### Luis Carlos Quesada, Mario Viquez, Daniel Ramirez - 5/19/2022
Ya despues de dormir hice algunas consultas y pareciera que podría ser la función de error, se me recomienda usar una biblioteca a la que esté más familiarizada y a la que se le pueda dar un contról más fino como PyTorch, podría sentarme a hacer un thinkering al error de la red anterior pero el coste de tiempo sería significativamente alto, por lo que mejor voy a crear la red nuevamente usando un modelo de torch, ya que la artquitetura está definida no debería tomar más de una hora.

Durante el día Gatto me dijo que es posible que no todos los datos sean del mismo tamaño, así que me interesa realizar un analisis preeliminar de los datos mas profundo que el anterior, ya que la vez pasada realmente no sabíamos mucho más que las distribuciones de datos y las composiciones de caracteristicas. Me interesa saber no solo los tamaños si no la posibilidad de una separación no supervisada con algun algoritmo tipo K-means simplemente para visualizar los puntos.

El día terminó siendo bastante productivo, hicimos algunas pruebas separando los datos con un PCA, K-means y graficando, salió terrible!
<div align="center"><img style="background: white;" src="Imgs\prueba-pca.png"></div>

Pero luego utilizando el modelo base hicimos un par de cambios: los datos ahora son pasados por un filtro que los escala y normaliza, luego las etiquetas fueron modificadas para hacer la predicción con one hot encoding en lugar de etiquetas ordinales.

### Resultados:
El modelo fue entrenado 700 epocas y alcanzó una precisión de 99%, lastimosamente esto es un obvio overfitting pues al revisar con el dataset de testing solo se alcanza un 46% de accuracy, aun así es mejor que el 20% de ayer.

En la tarde-malana del 21 esperamos aplicar finalmente el entrenamiento descrito en el paper, utilizando cross validation. 

---


