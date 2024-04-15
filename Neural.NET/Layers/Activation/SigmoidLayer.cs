using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Activation
{
    /// <summary>
    /// 標準シグモイド関数
    /// </summary>
    public class SigmoidLayer : IActivationLayer
    {
        DenseMatrix? output;
        readonly bool SAVE_OUTPUT_REF;

        public SigmoidLayer() : this(false) { }

        public SigmoidLayer(bool saveOutputRef) => this.SAVE_OUTPUT_REF = saveOutputRef;

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y)
        {
            x.PointwiseSigmoid(y);
            SaveOutput(y);
            return y;
        }

        public DenseMatrix Forward(DenseMatrix x)
        {
            var y = DenseMatrix.Create(x.RowCount, x.ColumnCount, 0.0f);
            Forward(x, y);
            return y;
        }

        public DenseMatrix Backward(DenseMatrix dOutput, DenseMatrix res)
        {
            if (this.output is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            this.output.Negate(res);
            res.Add(1.0f, res);
            res.PointwiseMultiply(this.output, res);
            res.PointwiseMultiply(dOutput, res);
            return res;
        }

        public DenseMatrix Backward(DenseMatrix dOutput)
        {
            if (this.output is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            var res = DenseMatrix.Create(this.output.RowCount, this.output.ColumnCount, 0.0f);
            return Backward(dOutput, res);
        }

        void SaveOutput(DenseMatrix output)
        {
            if (this.SAVE_OUTPUT_REF)
            {
                this.output = output;
                return;
            }

            this.output = output.CopyToOrClone(this.output);
        }
    }
}
