using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Activation
{
    /// <summary>
    /// ソフトマックス関数
    /// </summary>
    public class SoftmaxLayer : IActivationLayer
    {
        DenseMatrix? output;
        readonly bool SAVE_OUTPUT_REF;

        public SoftmaxLayer() : this(false) { }

        public SoftmaxLayer(bool saveOutputRef) => this.SAVE_OUTPUT_REF = saveOutputRef;

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y)
        {
            x.ColumnSoftmax(y);
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

            this.output.PointwiseMultiply(dOutput, res);
            var colSums = (DenseVector)res.ColumnSums();
            dOutput.SubtractRowVector(colSums, res);
            res.PointwiseMultiply(this.output, res);
            return res;
        }

        public DenseMatrix Backward(DenseMatrix dOutput)
        {
            if (this.output is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            var res = DenseMatrix.Create(dOutput.RowCount, dOutput.ColumnCount, 0.0f);
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
