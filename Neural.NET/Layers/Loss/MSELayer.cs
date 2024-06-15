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

            return this.loss.AsColumnMajorArray().Sum() / this.loss.ColumnCount;
        }

        public DenseMatrix Backward(DenseMatrix res)
        {
            if(this.diff is null || this.loss is null)
                throw Exceptions.CreateBackwardBeforeForwardException();

            if (res.RowCount != this.diff.RowCount || res.ColumnCount != this.diff.ColumnCount)
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(res), this.diff.RowCount.ToString(), this.diff.ColumnCount.ToString());


            this.diff.Multiply(2.0f / this.diff.ColumnCount, res);
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
