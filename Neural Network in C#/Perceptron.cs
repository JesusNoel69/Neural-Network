
namespace Neural_Network_in_C_
{
    public class Perceptron
    {
        private float[] inputs;
        public float[] weights;
        private float bias;
        private float learningRate;
        private string typeActivationFunction;
        public float output;
        public int expectedOutput; // Corrige el nombre aquí
        public float error = 0.0f;

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
            Console.WriteLine("salida esperada: " + expectedOutput);
            foreach (var weight in weights)
            {
                Console.Write("pesos: " + weight + " ");
            }
            Console.WriteLine("");
        }
        public Perceptron(int inputLength, string typeActivationFunction = "step", float learningRate = 0.1F)
        {
            this.learningRate = learningRate;
            bias = RandomizeNumber(1)[0];
            weights = RandomizeNumber(inputLength);
            this.typeActivationFunction = typeActivationFunction;
        }

        public void SetInputs(float[] inputs)
        {
            this.inputs = inputs;
        }

        public void SetExpectedOutput(int expectedOutput)
        {
            this.expectedOutput = expectedOutput;
        }

        public void CalculateOutput()
        {
            output = ActivationFunction(typeActivationFunction);
        }

        public void CalculateError()
        {
            error = expectedOutput - output;
        }

        public void UpdateWeights()
        {
            for (int eachInput = 0; eachInput < inputs.Length; eachInput++)
            {
                weights[eachInput] += learningRate * error * inputs[eachInput];
            }
            bias += learningRate * error;
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
                "thanh" => Thanh(),
                "relu" => ReLU(),
                _ => StepFunction(),
            };
        }

        public int StepFunction() => WeightedSum() >= 0 ? 1 : 0;

        public float SigmoidFunction()
        {
            float x = WeightedSum();
            return (float)(1 / (1 + Math.Exp(-x)));
        }

        public float Thanh()
        {
            float x = WeightedSum();
            return (float)((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
        }

        public float ReLU()
        {
            float x = WeightedSum();
            return x > 0 ? x : 0;
        }
    }
}
