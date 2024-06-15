using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// 2値クロスエントロピー誤差
    /// </summary>
    public class BinaryCrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? y;
        DenseMatrix? t;
        DenseMatrix? loss;
        DenseMatrix? negY;
        DenseMatrix? negT;

        public BinaryCrossEntropyLayer() : this(false) { }

        public BinaryCrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(y, t);

            if(this.loss is null || this.loss.RowCount != y.RowCount || this.loss.ColumnCount != y.ColumnCount)
                this.loss = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.negY is null || this.negY.RowCount != y.RowCount || this.negY.ColumnCount != y.ColumnCount)
                this.negY = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.negT is null || this.negT.RowCount != t.RowCount || this.negT.ColumnCount != t.ColumnCount)
                this.negT = DenseMatrix.Create(t.RowCount, t.ColumnCount, 0.0f);

            // t * log(y)
            y.PointwiseLog(this.loss);
            this.loss.PointwiseMultiply(t, this.loss);

            // log(1 - y)
            y.Subtract(1.0f, this.negY);
            this.negY.Negate(this.negY);
            this.negY.PointwiseLog(this.negY);

            // 1 - t
            t.Subtract(1.0f, this.negT);
            this.negT.Negate(this.negT);

            // -t * log(y) - (1 - t) * log(1 - y)
            this.negT.PointwiseMultiply(this.negY, this.negT);
            this.loss.Add(this.negT, this.loss);
            
            return -1.0f / y.ColumnCount * this.loss.AsColumnMajorArray().Sum(); 
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.y is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            Debug.Assert(this.negY is not null);
            Debug.Assert(this.negT is not null);

            // - t / y
            this.t.PointwiseDivide(this.y, res);
            res.Negate(res);

            // 1 - t
            this.t.Subtract(1.0f, this.negT);
            this.negT.Negate(this.negT);

            // 1 - y
            this.y.Subtract(1.0f, this.negY);
            this.negY.Negate(this.negY);

            // -t / y + (1 - t) / (1 - y) 
            this.negT.PointwiseDivide(this.negY, this.negT);
            res.Add(this.negT, res);
            res.Multiply(1.0f / y.ColumnCount, res);

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