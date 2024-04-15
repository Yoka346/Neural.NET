using System;
using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

using NeuralNET;
using MathNet.Numerics;
using System.Threading.Tasks;



static void Foo()
{
    const float EPSILON = 1.0e-4f;
    var x = Random.Shared.NextSingle() * 100.0f;
    var f0 = MathF.Log(x - EPSILON);
    var f1 = MathF.Log(x + EPSILON);
    Console.WriteLine($"Numerical: {(f1 - f0) / (2.0f * EPSILON)}");
    Console.WriteLine($"Analytic: {1.0f / x}");
}