
using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Activation
{
    /// <summary>
    /// ReLU関数
    /// </summary>
    public class ReLULayer : IActivationLayer
    {
        DenseMatrix? output;
        readonly bool SAVE_OUTPUT_REF;

        public ReLULayer() : this(false) { }

        public ReLULayer(bool saveOutputRef) => this.SAVE_OUTPUT_REF = saveOutputRef;

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y)
        {
            x.PointwiseMaximum(0.0f, y);
            SaveOutput(y);
            return y;
        }

        public DenseMatrix Forward(DenseMatrix x)
        {
            var y = DenseMatrix.Create(x.RowCount, x.ColumnCount, 0.0f);
            Forward(x, y);
            SaveOutput(y);
            return y;
        }

        public DenseMatrix Backward(DenseMatrix dOutput, DenseMatrix res)
        {
            if (this.output is null)
                throw new InvalidOperationException("Backward method must be called after forward.");

            this.output.PointwiseSign(res);
            res.PointwiseMultiply(dOutput);
            return res;
        }

        public DenseMatrix Backward(DenseMatrix dOutput)
        {
            if (this.output is null)
                throw new InvalidOperationException("Backward method must be called after forward.");

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
