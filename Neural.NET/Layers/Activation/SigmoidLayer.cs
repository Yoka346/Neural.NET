using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Activation
{
    /// <summary>
    /// 標準シグモイド関数
    /// </summary>
    public class SigmoidLayer : IActivationLayer
    {
        DenseMatrix? output;
        bool SaveOutputRef;

        public SigmoidLayer() : this(false) { }

        public SigmoidLayer(bool saveOutputRef) => this.SaveOutputRef = saveOutputRef;

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y)
        {
            x.Negate(y);
            y.PointwiseExp(y);
            y.Add(1.0f, y);
            y.Divide(1.0f, y);
            return y;
        }

        public DenseMatrix Forward(DenseMatrix x)
        {
            var y = (DenseMatrix)x.Clone();
            return Forward(x, y);
        }

        public DenseMatrix Backward(DenseMatrix dOutput, DenseMatrix res)
        {
            throw new NotImplementedException();
        }

        public DenseMatrix Backward(DenseMatrix dOutput)
        {
            throw new NotImplementedException();
        }
    }
}
