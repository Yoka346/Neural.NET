using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET.Layers.Loss
{
    /// <summary>
    /// 平均2乗誤差
    /// </summary>
    public class MSELayer : ILossLayer
    {
        DenseMatrix? loss;
        DenseMatrix? diff;

        public float Forward(DenseMatrix y, DenseMatrix t)
        {
            if(this.diff is null || this.diff.RowCount != y.RowCount || this.diff.ColumnCount != y.ColumnCount)
                this.diff = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            if(this.loss is null || this.loss.RowCount != y.RowCount || this.loss.ColumnCount != y.ColumnCount)
                this.loss = DenseMatrix.Create(y.RowCount, y.ColumnCount, 0.0f);

            y.Subtract(t, this.diff);
            this.diff.PointwiseMultiply(this.diff, this.loss);

            var lossArray = this.loss.AsColumnMajorArray();
            return lossArray.Sum() / lossArray.Length;
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.diff is null || this.loss is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            this.diff.Multiply(2.0f, res);
            res.Divide(this.diff.AsColumnMajorArray().Length, res);
            return res;
        }

        public DenseMatrix Backward()
        {
            if(this.diff is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            var res = DenseMatrix.Create(this.diff.RowCount, this.diff.ColumnCount, 0.0f);
            return Backward(res);
        }
    }
}
