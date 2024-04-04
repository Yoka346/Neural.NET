using System;
using System.Diagnostics;

using MathNet.Numerics.Providers.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Providers.OpenBLAS.LinearAlgebra;
using MathNet.Numerics.Providers.OpenBLAS;

const bool USE_MANAGED = true;

const int NUM_ROWS = 4096;
const int NUM_COLS = 4096;

var rand = new ContinuousUniform();
var lhs = DenseMatrix.CreateRandom(NUM_ROWS, NUM_COLS, rand);
var rhs = DenseMatrix.CreateRandom(NUM_ROWS, NUM_COLS, rand);
var product = DenseMatrix.Create(NUM_ROWS, NUM_COLS, 0.0f);

if(USE_MANAGED)
    LinearAlgebraControl.UseManaged();
else if(LinearAlgebraControl.TryUseNativeOpenBLAS())
    Console.WriteLine("Use OpenBLAS");

var sw = new Stopwatch();

sw.Start();
lhs.Multiply(rhs, product);
sw.Stop();

Console.WriteLine($"{sw.ElapsedMilliseconds}[ms]");
