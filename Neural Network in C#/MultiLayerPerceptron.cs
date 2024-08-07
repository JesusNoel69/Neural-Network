namespace Neural_Network_in_C_
{
     public class MultiLayerPerceptron
    {
        public List<List<Perceptron>> hiddenLayers { get; set; }
        public int epochs { get; set; }
        public List<Perceptron> outputLayer { get; set; }

        public MultiLayerPerceptron(List<List<Perceptron>> hiddenLayers, List<Perceptron> outputLayer, int epochs)
        {
            this.hiddenLayers = hiddenLayers;
            this.outputLayer = outputLayer;
            this.epochs = epochs;
        }

        public void Train(float[][] inputs, float[] expectedOutput)
        {
            for (int epoch = 0; epoch < epochs; epoch++) // iteraciones del entrenamiento
            {
                if(epoch%10==0){
                    Console.WriteLine($"Epoch {epoch + 1}");
                }

                for (int i = 0; i < inputs.Length; i++) // [0,0],[0,1],[1,0],[1,1]
                {
                    ForwardPass(inputs[i]);
                    BackwardPass(expectedOutput[i]);
                    if (epoch % 10 == 0) // cada diez epocas
                    {
                        foreach (Perceptron perceptron in outputLayer)
                        {
                            perceptron.Show();
                        }
                    }
                }
            }
        }

        public void ForwardPass(float[] inputs)
        {
            float[] currentInputs = inputs;

            foreach (List<Perceptron> eachLayer in hiddenLayers)
            {
                float[] layerOutputs = new float[eachLayer.Count];
                for (int each = 0; each < eachLayer.Count; each++) // numerro de neuronas por capa
                {
                    Perceptron perceptron = eachLayer[each];
                    perceptron.inputs = currentInputs; // Conectar salidas de la capa anterior

                    perceptron.output = perceptron.ActivationFunction(perceptron.typeActivationFunction);
                    layerOutputs[each] = perceptron.output; // guardar salida de la neurona actual
                }
                currentInputs = layerOutputs; // actualizar entradas para la siguiente capa
            }

            // Procesar la capa de salida
            foreach (Perceptron perceptron in outputLayer)
            {
                perceptron.inputs = currentInputs; // entradas para la capa de salida
                perceptron.output = perceptron.ActivationFunction(perceptron.typeActivationFunction);
            }
        }

        public void BackwardPass(float expectedOutput)
        {
            // Capa de salida
            foreach (Perceptron perceptron in outputLayer)
            {
                perceptron.espectedOutput = expectedOutput; // Salida esperada
                perceptron.error = perceptron.espectedOutput - perceptron.output;
                perceptron.UpdateWeights(perceptron.inputs.Length);
            }

            // Capas ocultas
            for (int i = hiddenLayers.Count - 1; i >= 0; i--)
            {
                List<Perceptron> layer = hiddenLayers[i];
                float[] errors = new float[layer.Count];

                //ultima capa
                if (i == hiddenLayers.Count - 1)
                {
                    for (int eachLayer = 0; eachLayer < layer.Count; eachLayer++)
                    {
                        float error = 0.0f;
                        foreach (Perceptron perceptron in outputLayer)
                        {
                            error += perceptron.error * perceptron.weights[eachLayer];
                        }
                        errors[eachLayer] = error;
                    }
                }
                //
                else
                {
                    List<Perceptron> nextLayer = hiddenLayers[i + 1];
                    for (int eachLayer = 0; eachLayer < layer.Count; eachLayer++)
                    {
                        float error = 0.0f;
                        for (int next = 0; next < nextLayer.Count; next++)
                        {
                            error += nextLayer[next].error * nextLayer[next].weights[eachLayer];
                        }
                        errors[eachLayer] = error;
                    }
                }

                for (int eachlayer = 0; eachlayer < layer.Count; eachlayer++)
                {
                    Perceptron perceptron = layer[eachlayer];
                    perceptron.error = errors[eachlayer];
                    perceptron.UpdateWeights(perceptron.inputs.Length);
                }
            }
        }
    }
}