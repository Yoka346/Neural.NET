using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// one-hotベクトルが正解データであるときに用いるクロスエントロピー誤差とソフトマックス関数を融合した層
    /// </summary>
    public class SoftmaxWithCategoricalCrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? t;
        DenseMatrix? expY;
        DenseVector? logSoftmax;

        public SoftmaxWithCategoricalCrossEntropyLayer() : this(false) { }

        public SoftmaxWithCategoricalCrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(t);

            if(this.expY is null || this.expY.RowCount != y.RowCount || this.expY.ColumnCount != y.ColumnCount)
                this.expY = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            y.PointwiseExp(this.expY);
            this.logSoftmax= (DenseVector)this.expY.ColumnSums();
            this.logSoftmax.PointwiseLog(this.logSoftmax);

            var entropy = 0.0f;
            for(var i = 0; i < y.ColumnCount; i++)
            {
                this.logSoftmax[i] = y[(int)t[0, i], i] - this.logSoftmax[i];
                entropy += this.logSoftmax[i];
            }
            return -entropy / y.ColumnCount;
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.logSoftmax is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            res.Clear();
            var nInv = 1.0f / this.logSoftmax.Count;
            for(var i = 0; i < this.logSoftmax.Count; i++)
            {
                var tIdx = (int)this.t[0, i]; 
                res[tIdx, i] = -nInv * (1.0f - MathF.Exp(this.logSoftmax[i]));
            }

            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.expY is null || this.logSoftmax is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            var res = DenseMatrix.Create(this.expY.RowCount, this.expY.ColumnCount, 0.0f);
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