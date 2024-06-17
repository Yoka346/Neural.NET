using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// クロスエントロピー誤差
    /// </summary>
    public class SigmoidWithBinaryCrossEntropyLayer : ILossLayer
    {
        readonly bool SAVE_INPUT_REF;
        DenseMatrix? t;
        DenseMatrix? sigY;
        DenseMatrix? loss;
        DenseMatrix? negY;
        DenseMatrix? negT;

        public SigmoidWithBinaryCrossEntropyLayer() : this(false) { }

        public SigmoidWithBinaryCrossEntropyLayer(bool saveInputRef) => this.SAVE_INPUT_REF = saveInputRef;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            SaveInput(t);

            if(this.loss is null || this.loss.RowCount != y.RowCount || this.loss.ColumnCount != y.ColumnCount)
                this.loss = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.sigY is null || this.sigY.RowCount != y.RowCount || this.sigY.ColumnCount != y.ColumnCount)
                this.sigY = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.negY is null || this.negY.RowCount != y.RowCount || this.negY.ColumnCount != y.ColumnCount)
                this.negY = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.negT is null || this.negT.RowCount != t.RowCount || this.negT.ColumnCount != t.ColumnCount)
                this.negT = DenseMatrix.Create(t.RowCount, t.ColumnCount, 0.0f);

            // apply sigmoid to y
            y.PointwiseSigmoid(this.sigY);

            // t * log(y)
            this.sigY.PointwiseLog(this.loss);
            this.loss.PointwiseMultiply(t, this.loss);

            // log(1 - y)
            this.sigY.Subtract(1.0f, this.negY);
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
            if(this.sigY is null || this.t is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            this.sigY.Subtract(this.t, res);
            res.Multiply(1.0f / this.sigY.ColumnCount, res);
            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.t is null)
                throw new Exception("Backward method must be called after forward.");

            var res = DenseMatrix.Create(this.t.RowCount, this.t.ColumnCount, 0.0f);
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