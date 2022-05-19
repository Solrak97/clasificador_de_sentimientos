# Bitacora: detección de sentimientos utilizando redes neuronales
### Investigadores:
* Luis Carlos Quesada
* Mario Viquez
* Gianfranco Bagnarello
* Isaac Herrera
* Daniel Ramirez
---
 
### Indice:

* [Replicación de la topología base](#1)
* [Busqueda de los datos de entrada al modelo propuesto](#2)

---

## <a mame="#1"> Replicación de la topología base Issa et al. </a>
### Luis Carlos Quesada - 5/10/2022
Repliqué el modelo de Issa et al. El modelo está descrito en el paper pero describe la capa de entrada como 193 neuronas, anteriormente habíamos hecho extracción de caracteristicas pero el resultado era de más de 228 variables a diferencia de este, por lo que no podemos comprobar de momento que la red neuronal se ajuste a las predicciones descritas en el paper.

### Resultados:
Se obtuvo un modelo con la topología descrita en el paper a replicar, sin embargo no se puede comparar los resultados ya que es necesario investigar la dimensionalidad de los datos de entrada pues no fueron descritos en el modelo original. 

<br>
<br>

## <a mame="#2"> Busqueda de los datos de entrada al modelo propuesto </a>
### Luis Carlos Quesada - 5/18/2022
Para poder hacer una clasificación de audio usando la topología de red propuesta en el paper es necesario saber las dimensiones de entrada de los datos, la red tiene 193 variables de entrada y sabemos que hay algunass caracteristicas con tamaños fijos.

Se sabe que el tonnetz es de shape (n, 6) entonces al aplicar el calculo de medias el shape final es de (6,).

Chromagram produce 12 bins default, pero el tamaño de bins puede ser cualquier numero > 0.

El MFCC usualmente tiene entre 20-40ms por cada bin, sabemos que tenemos 3 segundos por audio, entonces podemos mover eta variabe en el rango [75, 120].

Mel Spectrogram de la misma forma depende de la cantidad de mels, que de nuevo es [75, 120], el tamaño de MFCC y spectrogram deben ser iguales.

Contrast se calcula con 6 bandas default pero funciona con n_bands > 1 y retorna n_bands + 1

Así que al final, tenemos algo así

$$ 
6 + (bands + 1) + bins + 2mel = 193
$$ 
$$
mel ∈ [75, 120]
$$
$$
bands > 0
$$
$$
bins > 0
$$

### Resultados:
