using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using NeuralNET;
using NeuralNET.Layers.Loss;

namespace NeuralNETTests.Layers.Loss;

[TestClass]
public class SoftmaxWithCategoricalCrossEntropyLayerTests
{
    [TestMethod]
    public void BackwardTest()
    {
        const int NUM_ROWS = 4;
        const int NUM_COLS = 3;

        test(NUM_ROWS, NUM_COLS, true);
        test(NUM_COLS, NUM_ROWS, true);

        test(NUM_ROWS, NUM_COLS, false);
        test(NUM_COLS, NUM_ROWS, false);

        void test(int numRows, int numCols, bool saveInputRef)
        {
            var rand = new ContinuousUniform();
            var y = DenseMatrix.CreateRandom(numRows, numCols, rand);

            var t = DenseMatrix.Create(1, numCols, 0.0f);
            for(var i = 0; i < numCols; i++)
                t[0, i] = Random.Shared.Next(0, numRows);

            Utils.TestBackward(new SoftmaxWithCategoricalCrossEntropyLayer(saveInputRef), y, t);
        }
    }
}