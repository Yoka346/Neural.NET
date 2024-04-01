using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET.Layers;

namespace Neural.NET.Layers.Arithmetic
{
    /// <summary>
    /// 算術計算を行う層が実装するインターフェース.
    /// </summary>
    public interface IArithmeticLayer : ILayer
    {
        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y, DenseMatrix res);
        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y);

        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput, (DenseMatrix dx, DenseMatrix dy) res);
        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput);
    }
}
