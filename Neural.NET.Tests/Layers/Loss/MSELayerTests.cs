using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET.Layers.Loss;

namespace NeuralNETTests.Layers.Loss;

[TestClass]
public class MSELayerTests
{
    [TestMethod]
    public void BackwardTest()
    { 
        const int NUM_ROWS = 53;
        const int NUM_COLS = 47;
        
        test(NUM_ROWS, NUM_COLS);
        test(NUM_COLS, NUM_ROWS);

        void test(int numRows, int numCols)
        {
            var rand = new ContinuousUniform(-1, 1);
            var y = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var t = DenseMatrix.CreateRandom(numRows, numCols, rand);
            Utils.TestBackward(new MSELayer(), y, t);
        }
    }
}