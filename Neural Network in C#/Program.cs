using Neural_Network_in_C_;
/*
E1  E2  AND 
0   0   0
0   1   0
1   0   0
1   1   1
*/

// Definición de los datos de entrada y salida esperada
float[][] inputs =
[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

int[] expectedOutputs = [0, 0, 0, 1];

Layer inputLayer = new()
{
    layer = [new Perceptron(2), new Perceptron(2)]
};

List<Layer> hiddenLayers =
[
    new Layer { layer = [new Perceptron(2), new Perceptron(2)] },
    new Layer { layer = [new Perceptron(2), new Perceptron(2)] }
];

Layer outputLayer = new Layer
{
    layer = [new Perceptron(2)]
};
MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(inputLayer, hiddenLayers, outputLayer);
multiLayerPerceptron.TrainGlobal(inputs, expectedOutputs, 30); //inputs, expectedOutputs, epochs
