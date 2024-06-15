namespace NeuralNETTests.Layers.Loss;

using System.Diagnostics;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET.Layers.Loss;

public static class Utils
{
    /// <summary>
    /// 数値微分を計算する.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="input"></param>
    /// <returns></returns>
    public static DenseMatrix CalcNumericalGrad(ILossLayer layer, DenseMatrix y, DenseMatrix t)
    {
        const float EPSILON = 1.0e-3f;

        var grads = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);
        var gradsArray = grads.AsColumnMajorArray();
        var yArray = y.AsColumnMajorArray();
        for(var i = 0; i < yArray.Length; i++)
        {
            var tmp = yArray[i];

            yArray[i] = tmp - EPSILON;
            var f0 = layer.Forward(y, t);

            yArray[i] = tmp + EPSILON;
            var f1 = layer.Forward(y, t);

            yArray[i] = tmp;

            gradsArray[i] = (f1 - f0) / (2.0f * EPSILON);
            Debug.Assert(!float.IsNaN(gradsArray[i]));
        }

        return  grads;
    }

    /// <summary>
    /// 逆伝播のテストコード.
    /// </summary>
    public static void TestBackward(ILossLayer layer, DenseMatrix y, DenseMatrix t)
    {
        layer.Forward(y, t);
        var actual = layer.Backward();
        var expected = CalcNumericalGrad(layer, y, t);
        MatrixAssert.AreEqual(expected, actual);
    }
}