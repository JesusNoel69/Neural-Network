using Neural_Network_in_C_;
/*
E1  E2  AND 
0   0   0
0   1   0
1   0   0
1   1   1
*/

float[][] inputs = [[0,0],[0,1],[1,0],[1,1]]; // Datos de entrada
float[] expectedOutput = [0,0,0,1]; // Salida esperada

// Crear perceptrones para la capa oculta
List<List<Perceptron>> hiddenLayers = new List<List<Perceptron>>
{
    new List<Perceptron> { new Perceptron(2, "sigmoid"), new Perceptron(2,"sigmoid") },
    new List<Perceptron> { new Perceptron(2, "sigmoid"), new Perceptron(2, "sigmoid"), new Perceptron(2, "sigmoid"), new Perceptron(2,"sigmoid") },
    new List<Perceptron> { new Perceptron(4, "sigmoid"), new Perceptron(4,"sigmoid"), new Perceptron(4, "sigmoid"), new Perceptron(4,"sigmoid") },
};


// Crear perceptrones para la capa de salida (1 perceptrón para la salida final)
List<Perceptron> outputLayer = new List<Perceptron>
{
    new Perceptron(4, "step")
};

int epochs = 1000;

// Inicializar el MLP
MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(hiddenLayers, outputLayer, epochs);
multiLayerPerceptron.Train(inputs, expectedOutput);
