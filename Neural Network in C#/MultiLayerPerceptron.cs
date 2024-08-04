namespace Neural_Network_in_C_
{
    public class MultiLayerPerceptron
    {
        
        private Layer inputLayer;
        private List<Layer> hiddenLayers;
        private Layer  outputLayer;
        private bool globalError;

        public MultiLayerPerceptron(Layer  inputLayer, List<Layer> hiddenLayers, Layer  outputLayer){
            this.inputLayer = inputLayer;
            this.hiddenLayers = hiddenLayers;
            this.outputLayer = outputLayer;
        }
       public void TrainGlobal(float[][] trainingInputs, int[] trainingOutputs, int epochs)
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Época: {epoch}");
                for (int eachInput = 0; eachInput < trainingInputs.Length; eachInput++)
                {
                    ForwardPropagation(trainingInputs[eachInput]);
                    BackwardPropagation(trainingOutputs[eachInput]);
                    UpdateWeights();
                    ShowData(epoch);
                }
                Console.WriteLine("-------------------");
            }
        }
         private void ShowData(int epoch)
        {
            Console.WriteLine("Capa de salida:");
            outputLayer.layer.ForEach(perceptron => perceptron.Show());
        }
        private void ForwardPropagation(float[] inputs)
        {
            //capa de entrads
            for (int eachLayer = 0; eachLayer < inputLayer.layer.Count; eachLayer++)
            {
                inputLayer.layer[eachLayer].SetInputs(inputs);
                inputLayer.layer[eachLayer].CalculateOutput();
            }

            //capas ocultas
            for (int layerIndex = 0; layerIndex < hiddenLayers.Count; layerIndex++)
            {
                Layer currentLayer = hiddenLayers[layerIndex];
                float[] previousLayerOutputs = layerIndex == 0 
                    ? inputLayer.layer.Select(p => p.output).ToArray() 
                    : hiddenLayers[layerIndex - 1].layer.Select(p => p.output).ToArray();

                for (int eachLayer = 0; eachLayer < currentLayer.layer.Count; eachLayer++)
                {
                    currentLayer.layer[eachLayer].SetInputs(previousLayerOutputs);
                    currentLayer.layer[eachLayer].CalculateOutput();
                }
            }
            //

            // capa salida
            float[] finalHiddenLayerOutputs = hiddenLayers.Last().layer.Select(p => p.output).ToArray();
            for (int eachLayer = 0; eachLayer < outputLayer.layer.Count; eachLayer++)
            {
                outputLayer.layer[eachLayer].SetInputs(finalHiddenLayerOutputs);
                outputLayer.layer[eachLayer].CalculateOutput();
            }
        }

        private void BackwardPropagation(int expectedOutput)
        {
            // error de la capa de salida
            for (int eachOutput = 0; eachOutput < outputLayer.layer.Count; eachOutput++)
            {
                outputLayer.layer[eachOutput].SetExpectedOutput(expectedOutput);
                outputLayer.layer[eachOutput].CalculateError();
            }

            // porpagar para atras el error a travs de las capas ocultas
            for (int layerIndex = hiddenLayers.Count - 1; layerIndex >= 0; layerIndex--)
            {
                Layer currentLayer = hiddenLayers[layerIndex];
                float[] nextLayerErrors = layerIndex == hiddenLayers.Count - 1 
                    ? outputLayer.layer.Select(p => p.error).ToArray()
                    : hiddenLayers[layerIndex + 1].layer.Select(p => p.error).ToArray();

                for (int eachPerceptron = 0; eachPerceptron < currentLayer.layer.Count; eachPerceptron++)
                {
                    Perceptron perceptron = currentLayer.layer[eachPerceptron];
                    float errorSum = nextLayerErrors.Sum(e => e * perceptron.weights[eachPerceptron]);
                    perceptron.error = errorSum * perceptron.output * (1 - perceptron.output);
                }
            }
        }

        private void UpdateWeights()
        {
            // actualizar los pesos en todas las capas
            foreach (var eachlayer in hiddenLayers)
            {
                foreach (var perceptron in eachlayer.layer)
                {
                    perceptron.UpdateWeights();
                }
            }

            foreach (var perceptron in outputLayer.layer)
            {
                perceptron.UpdateWeights();
            }
        }

    }
}