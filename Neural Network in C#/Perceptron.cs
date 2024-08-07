namespace Neural_Network_in_C_
{
        public class Perceptron
    {
        public float[] inputs { get; set; }
        public float[] weights { get; set; }
        public float bias { get; set; }
        public float learningRate { get; set; }
        public string typeActivationFunction { get; set; }
        public float output { get; set; }
        public float espectedOutput { get; set; }
        public float error = 0.0f;

        public Perceptron(int inputLength, string typeActivationFunction = "step", float learningRate = 0.1F)
        {
            weights = RandomizeNumber(inputLength);
            bias = RandomizeNumber(1)[0];
            this.learningRate = learningRate;
            this.typeActivationFunction = typeActivationFunction;
        }

        public static float[] RandomizeNumber(int length)
        {
            Random random = new();
            float[] randomNumbers = new float[length];
            for (int eachInput = 0; eachInput < length; eachInput++)
            {
                float number = (float)(random.NextDouble() * 2 - 1);
                randomNumbers[eachInput] = number;
            }
            return randomNumbers;
        }

        public void Show()
        {
            Console.Write("Error: " + error + " ");
            Console.WriteLine("Sesgo: " + bias);
            Console.Write("salida: " + output + " ");
            Console.WriteLine("salida esperada: " + espectedOutput);
            foreach (var weight in weights)
            {
                Console.Write("pesos: " + weight + " ");
            }
            Console.WriteLine("");
        }

        public void UpdateWeights(int length)
        {
            for (int eachInput = 0; eachInput < length; eachInput++)
            {
                weights[eachInput] += learningRate * error * ActivationFunctionDerivative(output) * inputs[eachInput];
            }
            bias += learningRate * error * ActivationFunctionDerivative(output);
        }

        public float WeightedSum()
        {
            float sum = 0f;
            for (int eachInput = 0; eachInput < inputs.Length; eachInput++)
            {
                sum += inputs[eachInput] * weights[eachInput];
            }
            sum += bias;
            return sum;
        }

        public float ActivationFunction(string functionName)
        {
            return functionName switch
            {
                "step" => StepFunction(),
                "sigmoid" => SigmoidFunction(),
                "tanh" => Tanh(),
                "relu" => ReLU(),
                _ => StepFunction(),
            };
        }

        public float ActivationFunctionDerivative(float x)
        {
            return typeActivationFunction switch
            {
                "sigmoid" => SigmoidDerivative(x),
                "tanh" => TanhDerivative(x),
                "relu" => ReLUDerivative(x),
                _ => StepDerivative(),
            };
        }

        public int StepFunction() => WeightedSum() >= 0 ? 1 : 0;
        public float StepDerivative() => 1; //parametro x
        public float SigmoidFunction()
        {
            float x = WeightedSum();
            return (float)(1 / (1 + Math.Exp(-x)));
        }
        public float SigmoidDerivative(float x) => x * (1 - x);

        public float Tanh()
        {
            float x = WeightedSum();
            return (float)((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
        }
        public float TanhDerivative(float x) => 1 - x * x;

        public float ReLU()
        {
            float x = WeightedSum();
            return x > 0 ? x : 0;
        }
        public float ReLUDerivative(float x) => x > 0 ? 1 : 0;
    }


}
