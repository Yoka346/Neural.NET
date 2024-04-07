using System;
using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

using NeuralNET;
using MathNet.Numerics;

const int NUM_ROWS = 8195;
const int NUM_COLS = 8195;

var rand = new ContinuousUniform();

DenseMatrix lhs, res, resExpected;
DenseVector rhs;

var sw = new Stopwatch();

lhs = DenseMatrix.CreateRandom(NUM_ROWS, NUM_COLS, rand);
rhs = DenseVector.CreateRandom(NUM_ROWS, rand);
res = DenseMatrix.Create(NUM_ROWS, NUM_COLS, 0.0f);
resExpected = DenseMatrix.Create(NUM_ROWS, NUM_COLS, 0.0f);

lhs.SubtractRowVector(rhs, res);
SubtractRowVectorTest(lhs, rhs, resExpected);
Console.WriteLine(AreEqual(res, resExpected));

static bool AreEqual(DenseMatrix a, DenseMatrix b)
{
    if(a.RowCount != b.RowCount || a.ColumnCount != b.ColumnCount)
        return false;

    for(var i = 0; i < a.RowCount; i++)
        for(var j = 0; j < b.RowCount; j++)
            if(MathF.Abs(a[i, j] - b[i, j]) > 1.0e-7f)
                return false;
    return true;
}

static void SubtractRowVectorTest(DenseMatrix lhs, DenseVector rhs, DenseMatrix sum)
{
    for(var i = 0; i < lhs.ColumnCount; i++)
    {
        for(var j = 0; j < lhs.RowCount; j++)
            sum[j, i] = lhs[j, i] - rhs[i];        
    }
}