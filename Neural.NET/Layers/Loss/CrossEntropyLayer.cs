using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// クロスエントロピー誤差
    /// </summary>
    public class CrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? y;
        DenseMatrix? t;
        DenseMatrix? cache;

        public CrossEntropyLayer() : this(false) { }

        public CrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(y, t);

            if(this.cache is null || this.cache.RowCount != y.RowCount || this.cache.ColumnCount != y.ColumnCount)
                this.cache = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            y.PointwiseLog(this.cache);
            this.cache.PointwiseMultiply(t, this.cache);
            
            return -1.0f / y.ColumnCount * this.cache.AsColumnMajorArray().Sum(); 
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.y is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            this.t.PointwiseDivide(this.y, res);
            res.Multiply(-1.0f / this.y.ColumnCount, res);
            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.y is null)
                throw new Exception("Backward method must be called after forward.");

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