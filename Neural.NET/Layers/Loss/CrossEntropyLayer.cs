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
        DenseMatrix? loss;

        public CrossEntropyLayer() : this(false) { }

        public CrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(y, t);

            if(this.loss is null || this.loss.RowCount != y.RowCount || this.loss.ColumnCount != y.ColumnCount)
                this.loss = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            y.PointwiseLog(this.loss);
            this.loss.PointwiseMultiply(t, this.loss);
            
            return -1.0f / y.ColumnCount * this.loss.AsColumnMajorArray().Sum(); 
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