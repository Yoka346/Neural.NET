namespace NeuralNETTests.Layers.Activation;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET.Layers.Activation;

public static class Utils
{
    /// <summary>
    /// 数値微分を計算する.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="input"></param>
    /// <returns></returns>
    public static DenseMatrix CalcNumericalGrad(IActivationLayer layer, DenseMatrix input)
    {
        const float EPSILON = 1.0e-4f;

        var epsilon = DenseMatrix.Create(input.RowCount, input.ColumnCount, EPSILON);
        return (DenseMatrix)((layer.Forward(input + epsilon) - layer.Forward(input - epsilon)) / (2.0f * EPSILON));
    }

    /// <summary>
    /// 逆伝播のテストコード.
    /// </summary>
    public static void TestBackward(IActivationLayer layer)
    {
        // 端数処理のバグも検知するために,行列の大きさは素数にする.
        int numRows = 4099;
        int numCols = 4093;
        var rand = new ContinuousUniform(-1.0, 1.0);
        test(numRows, numCols);
        test(numCols, numRows);

        void test(int numRows, int numCols)
        {
            var input = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var dOutput = DenseMatrix.CreateRandom(numRows, numCols, rand);

            layer.Forward(input);
            var actual = layer.Backward(dOutput);

            var expected = (DenseMatrix)CalcNumericalGrad(layer, input).PointwiseMultiply(dOutput);

            MatrixAssert.AreEqual(expected, actual);
        }
    }
}