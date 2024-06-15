namespace NeuralNETTests;

using MathNet.Numerics.LinearAlgebra.Single;

internal static class MatrixAssert
{
    public static void AreEqual(DenseMatrix expected, DenseMatrix actual, float epsilon = 1.0e-2f)
    {
        Assert.AreEqual(expected.RowCount, actual.RowCount);
        Assert.AreEqual(expected.ColumnCount, actual.ColumnCount);

        var expectedArray = expected.AsColumnMajorArray();
        var actualArray = actual.AsColumnMajorArray();
        for(var i = 0; i < expectedArray.Length; i++)
            Assert.AreEqual(expectedArray[i], actualArray[i], epsilon);
    }
}