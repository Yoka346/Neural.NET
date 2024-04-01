using MathNet.Numerics.LinearAlgebra.Single;

using NeuralNET;

namespace NeuralNET.Layers.Arithmetic
{
    public class MultiplyLayer : IArithmeticLayer
    {
        DenseMatrix? x;
        DenseMatrix? y;
        bool saveRef;

        public MultiplyLayer() : this(false) { }

        /// <summary>
        /// 乗算層のインスタンスを生成します.
        /// </summary>
        /// <param name="saveInputRef">この層への入力の参照を保持するかどうか</param>
        public MultiplyLayer(bool saveInputRef) => this.saveRef = saveInputRef;

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y, DenseMatrix res)
        {
            SaveInput(x, y);
            x.PointwiseMultiply(y, res);
            return res;
        }

        public DenseMatrix Forward(DenseMatrix x, DenseMatrix y) 
        { 
            SaveInput(x, y);
            return (DenseMatrix)x.PointwiseMultiply(y); 
        }

        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput, (DenseMatrix dx, DenseMatrix dy) res)
        {
            dOutput.PointwiseMultiply(this.y, res.dx);
            dOutput.PointwiseMultiply(this.x, res.dy);
            return res;
        }

        public (DenseMatrix dx, DenseMatrix dy) Backward(DenseMatrix dOutput)
        {
            var dx = (DenseMatrix)dOutput.PointwiseMultiply(this.y);
            var dy = (DenseMatrix)dOutput.PointwiseMultiply(this.x);
            return (dx, dy);
        }

        void SaveInput(DenseMatrix x, DenseMatrix y)
        {
            if (this.saveRef)
            {
                (this.x, this.y) = (x, y);
                return;
            }

            this.x = x.CopyToOrClone(this.x);
            this.y = y.CopyToOrClone(this.y);
        }
    }
}
