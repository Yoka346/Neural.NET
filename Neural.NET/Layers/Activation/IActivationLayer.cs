﻿using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET.Layers;

namespace Neural.NET.Layers.Activation
{
    /// <summary>
    /// 活性化層(活性化関数)が実装するインターフェース
    /// </summary>
    public interface IActivationLayer : ILayer
    {
        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y);
        public DenseMatrix Forward(DenseMatrix x);

        public DenseMatrix Backward(DenseMatrix dOutput, DenseMatrix res);
        public DenseMatrix Backward(DenseMatrix dOutput);
    }
}