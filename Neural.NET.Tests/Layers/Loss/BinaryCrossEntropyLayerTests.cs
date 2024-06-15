using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using NeuralNET;
using NeuralNET.Layers.Loss;

namespace NeuralNETTests.Layers.Loss;

[TestClass]
public class BinaryCrossEntropyLayerTests
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
            var rand = new ContinuousUniform(0, 1);
            var y = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var t = DenseMatrix.CreateRandom(numRows, numCols, rand);
            Utils.TestBackward(new BinaryCrossEntropyLayer(saveInputRef), y, t);
        }
    }
}