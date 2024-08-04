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
        public float output;
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
        public Perceptron(int inputLength, string typeActivationFunction="step", float learningRate=0.1F)
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
        public float ActivationFunction(string functionName){

            return functionName switch
            {
                "step" => StepFunction(),
                "sigmoid" => SigmoidFunction(),
                "thanh" => Thanh(),
                "relu" => ReLU(),
                _ => StepFunction(),
            };
        }
        //1 o 0
        public int StepFunction()=> WeightedSum()>=0? 1 : 0;
        //mapea el valor entre 0 y 1
        public float SigmoidFunction(){
            float x = WeightedSum();
            return 1/(1+Math.Exp(-x));
        }
        //mapea el valor entre -1 y 1 
        public float Thanh(){
            float x = WeightedSum();
            return (Math.Exp(x)-Math.Exp(-x))/(Math.Exp(x)+Math.Exp(-x));
        }
        //positivo devuelve el valor sino devuelve 0
        public float ReLU(){ 
            float x = weightedSum();
            return x>0? x: 0;
        }
    }
}