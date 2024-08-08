# Red Neuronal implementada desde cero
## Descripcion
Se implemento una red neuronal (mas especificamente una red neuronal de alimentacion directa usando capas ocultas) basica en C# con el objetivo de profundizar en este tema de redes neuronales y ver todos los componentes lógicos que trae consigo.

**Al ser un proyecto dedicado al aprendizaje puede estar sujeto a prácticas poco convencionales.**

## Uso:
navega al directorio con: 
```
cd directorio
```
corre el proyecto usando:
```
dotnet run
```
## Requisitos:
.Net 8 o superior

## Definiciones consideradas:
### Perceptrón:
Es una red neuronal sencilla que contiene una cantidad de entradas (caracteristicas), cada entrada contiene un peso determinado que no es mas que un valor dado para cada entrada, contiene una funcion de activacion con salida binaria (0 ó 1) y su salida que es la suma ponderada (sumatoria de{ pesos<sub>i</sub> * entradas<sub>i</sub>}), sumando un sesgo a la salida de la suma y aplicando una funcion de activacion (en caso de un perceptron simple una binaria resultado de la suma ponderada mayor o igual 0 devuelve 1 si es menor a 0 devuelve 0).

Salida= i...n( peso<sub>i</sub> * entrada<sub>i</sub> ) + sesgo
Step = f(x)= {1 si x >= 0, 0 si x < 0}


### Perceptrón multicapa 

Es un tipo de red neuronal formada por multiples capas, capa de entrada, capas ocultas y capa de salida, en el cual cada capa contiene un numero variable de perceptrones sin embargo, los perceptrones tendran mas funciones de activacion (ReLU, Sigmoid y TanH las mas comunes) y para dar retorlimentacion de los errores las derivadas de dichas funciones, las salidas de cada capa son las entradas de la siguiente capa; La capa de entrada no procesa los datos de ninguna manera, solo los pasa a la primer capa de las capas ocultas, las capas ocultas randomizan los datos de la capa anterior y los procesan tal cual para su salida pero sin calcular ningun error, la capa de salida calcula el error y realiza una retroalimentacion inversa a las capas anteriores para recalcular los valores hasta que el error sea nulo o aceptable, y se repite de esta forma una cantidad de veces determinadas llamadas epocas.

Sigmoid = σ(x) = 1/(1 + e<sup>-x</sup>)

ReLU = ReLU(x) = {x si x > 0, 0 si x <= 0}

TanH = tanh(x) = (e<sup>x</sup>-e<sup>-x</sup>)/(e<sup>x</sup>+e<sup>-x</sup>)

Sigmoid` = σ`(x) = σ(x)*(1-σ(x))

ReLU` = ReLU`(x) = {x si x > 0, 0 si x <= 0}

TanH` = tanh`(x) = 1-tanh<sup>2</sup>(x)
