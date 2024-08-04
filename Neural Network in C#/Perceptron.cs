using System.Runtime.InteropServices;

namespace Neural_Network_in_C_
{
    public class Perceptron
    {
        private float[] inputs;
        private float[] weights=[];
        private float bias;
        private float learningRate;
        private string typeActivationFunction;
        public int output;
        public int espectedOutput;
        private float error = 0.0f;
        public static float[] RandomizeNumber(int length){
            Random random = new();
            float[] randomNumbers=new float[length];
            for(int eachInput=0; eachInput<length; eachInput++){
                float number = (float)(random.NextDouble()*2-1);
                randomNumbers[eachInput] = number;
            }
            return randomNumbers;
        }
        public Perceptron(int inputLength, string typeActivationFunction="scalon", float learningRate=0.1F)
        {
            this.learningRate = learningRate;
            bias = RandomizeNumber(1)[0];
            weights = RandomizeNumber(inputLength);
            this.typeActivationFunction = typeActivationFunction;

        }

        public void Train(float[][] trainingInputs, int[] trainingOutputs, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                bool accept = true;
                System.Console.WriteLine("-------------");
                Console.WriteLine("iteracion: "+(epoch+1));
                for (int eachInput = 0; eachInput < trainingInputs.Length; eachInput++)
                {
                    inputs = trainingInputs[eachInput]; //[0,0]
                    espectedOutput = trainingOutputs[eachInput];//0
                    output = ActivationFunction(typeActivationFunction);
                    error =  espectedOutput-output;
                    if (error != 0)
                    {
                        accept = false;
                        UpdateWeights(inputs.Length);
                    }
                    Show();
                }
                System.Console.WriteLine("-------------");
                if (accept)
                {
                    break;
                }
            }
        }
        public void Show(){
            Console.Write("Error: "+error+" ");
            Console.WriteLine("Sesgo: "+bias);
            Console.Write("salida: "+output+" ");
            Console.WriteLine("salida esperada: "+espectedOutput);
            foreach(var weight in weights){
                Console.Write("pesos: "+weight+" ");
            }
            Console.WriteLine("");
        }
        public void UpdateWeights(int length){
            for(int eachInput=0; eachInput<length; eachInput++){
                weights[eachInput]+=learningRate*error*inputs[eachInput];
            }
            bias+=learningRate*error;
        }
        public float WeightedSum(){
            float sum = 0f;
            for(int eachInput=0; eachInput<inputs.Length; eachInput++){
                sum += inputs[eachInput]*weights[eachInput];
            }
            sum+=bias;
            return sum;
        }
        public int ActivationFunction(string functionName){

            return functionName switch
            {
                "scalon" => Scalon(),
                "sigmoide" => Scalon(),
                _ => Scalon(),
            };
        }
        public int Scalon()=> WeightedSum()>=0? 1 : 0;
    }
}