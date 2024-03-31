using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra.Single;
using NeuralNET.Layers;

namespace Neural.NET.Layers.Arithmetic
{
    /// <summary>
    /// 加算レイヤー
    /// </summary>
    public class AddLayer : IArithmeticLayer
    {
        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y, DenseMatrix res)
        {
            x.Add(y, res);
            return res;
        }

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y) => x + y;

        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput, (DenseMatrix dx, DenseMatrix dy) res)
        {
            dOutput.CopyTo(res.dx);
            dOutput.CopyTo(res.dy);
            return res;
        }

        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput) 
            => ((DenseMatrix)dOutput.Clone(), (DenseMatrix)dOutput.Clone());
    }
}
