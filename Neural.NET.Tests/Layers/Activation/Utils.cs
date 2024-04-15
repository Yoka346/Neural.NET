namespace NeuralNETTests.Layers.Activation;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;
using NeuralNET;
using NeuralNET.Layers.Activation;

public static class Utils
{
    /// <summary>
    /// 数値微分を計算する.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="input"></param>
    /// <returns></returns>
    public static DenseMatrix CalcPointwiseNumericalDiff(IActivationLayer layer, DenseMatrix input)
    {
        const float EPSILON = 1.0e-4f;

        var epsilon = DenseMatrix.Create(input.RowCount, input.ColumnCount, EPSILON);
        return (DenseMatrix)((layer.Forward(input + epsilon) - layer.Forward(input - epsilon)) / (2.0f * EPSILON));
    }

    /// <summary>
    /// 数値微分で勾配ベクトルを計算する.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="input"></param>
    /// <returns></returns>
    public static DenseMatrix CalcNumericalGradient(IActivationLayer layer, DenseMatrix input, DenseMatrix dOutput)
    {
        const float EPSILON = 1.0e-4f;

        var f0 = DenseMatrix.Create(input.RowCount, input.ColumnCount, 0.0f);
        var f1 = DenseMatrix.Create(input.RowCount, input.ColumnCount, 0.0f);
        var grads = DenseMatrix.Create(input.RowCount, input.ColumnCount, 0.0f);
        for(var i = 0; i < input.RowCount; i++)
        {
            input.CopyTo(f0);
            input.CopyTo(f1);
            for(var j = 0; j < input.ColumnCount; j++)
            {
                f0[i, j] = input[i, j] - EPSILON;
                f1[i, j] = input[i, j] + EPSILON;
            }

            layer.Forward(f0, f0);
            layer.Forward(f1, f1);

            var df = f1 - f0;
            df.Divide(2.0f * EPSILON, df);

            df.PointwiseMultiply(dOutput, df);

            var colSums = df.ColumnSums();
            for(var j = 0; j < input.ColumnCount; j++)
                grads[i, j] = colSums[j];
        }

        return grads;
    }

    /// <summary>
    /// 逆伝播のテストコード.
    /// </summary>
    public static void TestPointwiseBackward(IActivationLayer layer)
    {
        // 端数処理のバグも検知するために,行列の大きさは素数にする.
        int numRows = 1031;
        int numCols = 1019;
        var rand = new ContinuousUniform(-1.0, 1.0);
        test(numRows, numCols);
        test(numCols, numRows);

        void test(int numRows, int numCols)
        {
            var input = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var dOutput = DenseMatrix.CreateRandom(numRows, numCols, rand);

            layer.Forward(input);
            var actual = layer.Backward(dOutput);

            var expected = (DenseMatrix)CalcPointwiseNumericalDiff(layer, input).PointwiseMultiply(dOutput);

            MatrixAssert.AreEqual(expected, actual);
        }
    }

    /// <summary>
    /// 逆伝播のテストコード.
    /// </summary>
    public static void TestBackward(IActivationLayer layer)
    {
        // 端数処理のバグも検知するために,行列の大きさは素数にする.
        int numRows = 53;
        int numCols = 47;
        var rand = new ContinuousUniform(-1.0, 1.0);
        test(numRows, numCols);
        test(numCols, numRows);

        void test(int numRows, int numCols)
        {
            var input = DenseMatrix.CreateRandom(numRows, numCols, rand);
            var dOutput = DenseMatrix.CreateRandom(numRows, numCols, rand);

            layer.Forward(input);
            var actual = layer.Backward(dOutput);

            var expected = CalcNumericalGradient(layer, input, dOutput);

            MatrixAssert.AreEqual(expected, actual);
        }
    }
}