using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// 損失計算を行う層が実装するインターフェース.
    /// </summary>
    public interface ILossLayer : ILayer
    {
        public float Forward(DenseMatrix y, DenseMatrix t);
        public DenseMatrix Backward();
        public DenseMatrix Backward(DenseMatrix res);
    }
}
