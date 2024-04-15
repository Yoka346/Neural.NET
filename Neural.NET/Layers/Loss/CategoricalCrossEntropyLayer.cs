using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// one-hotベクトルが正解データであるときに用いるクロスエントロピー誤差
    /// </summary>
    public class CategoricalCrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? y;
        DenseMatrix? t;
        public CategoricalCrossEntropyLayer() : this(false) { }

        public CategoricalCrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(y, t);

            var entropy = 0.0f;
            for(var i = 0; i < y.ColumnCount; i++)
                entropy += MathF.Log(y[(int)t[0, i], i]);
            return -entropy / y.ColumnCount;
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.y is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            res.Clear();
            for(var i = 0; i < y.ColumnCount; i++)
            {
                var tIdx = (int)this.t[0, i]; 
                res[tIdx, i] = -1.0f / (y.ColumnCount * y[tIdx, i]);
            }

            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.y is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            var res = DenseMatrix.Create(this.y.RowCount, this.y.ColumnCount, 0.0f);
            return Backward(res);
        }

        void SaveInput(DenseMatrix y, DenseMatrix t)
        {
            if(this.SAVE_INPUT_REF)
            {
                (this.y, this.t) = (y, t);
                return;
            }

            this.y = y.CopyToOrClone(this.y);
            this.t = t.CopyToOrClone(this.t);
        }
    }
}