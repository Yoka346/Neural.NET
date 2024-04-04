using System;

using MathNet.Numerics.LinearAlgebra.Single;

var mat = DenseMatrix.Create(4, 4, 0.0f);
mat[0, 0] = 1.0f;
Console.WriteLine(mat.PointwiseSign());
