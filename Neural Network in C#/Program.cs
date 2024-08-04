using Neural_Network_in_C_;
/*
E1  E2  AND 
0   0   0
0   1   0
1   0   0
1   1   1
*/

// Perceptron perceptronSimple = new([1,0,0,1]);//inputs
// perceptronSimple.AcitvationFunction("scalon");
// Console.WriteLine(perceptronSimple.output);
//inputs[]  , expectedOutput , learnRate?
float[][] inputs =[[0,0],[0,1],[1,0],[1,1]];
int[] espectedOutput=[0,0,0,1];
Perceptron perceptronSimple = new(inputs[0].Length);
perceptronSimple.Train(inputs, espectedOutput, 30);