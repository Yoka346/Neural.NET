using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using NeuralNET;
using NeuralNET.Layers.Loss;

namespace NeuralNETTests.Layers.Loss;

[TestClass]
public class CrossEntropyLayerTests
{
    [TestMethod]
    public void BackwardTest()
    {
        const int NUM_ROWS =53;
        const int NUM_COLS = 47;

        test(NUM_ROWS, NUM_COLS, true);
        test(NUM_COLS, NUM_ROWS, true);

        test(NUM_ROWS, NUM_COLS, false);
        test(NUM_COLS, NUM_ROWS, false);

        void test(int numRows, int numCols, bool saveInputRef)
        {
            var rand = new ContinuousUniform();
            var y = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var t = DenseMatrix.CreateRandom(numRows, numCols, rand);
            y.DivideByRowVector((DenseVector)y.ColumnSums(), y);
            t.DivideByRowVector((DenseVector)t.ColumnSums(), t);
            Utils.TestBackward(new CrossEntropyLayer(saveInputRef), y, t);
        }
    }
}