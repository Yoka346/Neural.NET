using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// ソフトマックス関数 + クロスエントロピー誤差
    /// </summary>
    public class SoftmaxWithCrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? t;
        DenseMatrix? logSoftmax;
        DenseMatrix? loss;

        public SoftmaxWithCrossEntropyLayer() : this(false) { }

        public SoftmaxWithCrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(t);

            if(this.loss is null || this.loss.RowCount != y.RowCount || this.loss.ColumnCount != y.ColumnCount)
                this.loss = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.logSoftmax is null || this.logSoftmax.RowCount != y.RowCount || this.logSoftmax.ColumnCount != y.ColumnCount)
                this.logSoftmax = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            y.ColumnLogSoftmax(this.logSoftmax);
            this.logSoftmax.PointwiseMultiply(t, this.loss);
            
            return -1.0f / y.ColumnCount * this.loss.AsColumnMajorArray().Sum(); 
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.logSoftmax is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            this.logSoftmax.PointwiseExp(res);
            res.Subtract(t, res);
            res.Multiply(1.0f / this.logSoftmax.ColumnCount, res);
            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.logSoftmax is null)
                throw new Exception("Backward method must be called after forward.");

            var res = DenseMatrix.Create(this.logSoftmax.RowCount, this.logSoftmax.ColumnCount, 0.0f);
            return Backward(res);
        }

        void SaveInput(DenseMatrix t)
        {
            if(this.SAVE_INPUT_REF)
            {
                this.t = t;
                return;
            }

            this.t = t.CopyToOrClone(this.t);
        }
    }
}