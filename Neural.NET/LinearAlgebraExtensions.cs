using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET
{
    public static class LinearAlgebraExtensions
    {
        /// <summary>
        /// 密行列に密列ベクトルをブロードキャストして加算する.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <param name="sum"></param>
        public static void AddColumnVector(this DenseMatrix lhs, DenseVector rhs, DenseMatrix sum)
        {
            if (lhs.RowCount != rhs.Count)
                throw Exceptions.CreateInvalidVectorDimensionException(nameof(rhs), nameof(lhs.RowCount));

            if (!lhs.DimensionEqualsTo(sum))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(sum), nameof(lhs.RowCount), nameof(lhs.ColumnCount));

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var rhsArray = rhs.AsArray();
            var sumArray = sum.AsColumnMajorArray();
            var (numRows, numCols) = (sum.RowCount, sum.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var s = sumArray.AsSpan(j * numRows);
                var rest = numRows % UNROLL_STEP_NUM;
                var end = numRows - rest;
                for (var i = 0; i < end; i += UNROLL_STEP_NUM)
                {
                    var leftVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(l[i..]));
                    var rightVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(rhsArray.AsSpan(i)));
                    var sumVec = leftVec + rightVec;
                    sumVec.StoreUnsafe(ref MemoryMarshal.GetReference(s[i..]));
                }

                // 端数処理
                for(var i = end; i < numRows; i++)
                    s[i] = l[i] + rhsArray[i];
            });
        }

        /// <summary>
        /// 密行列に密行ベクトルをブロードキャストして加算する.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <param name="sum"></param>
        public static void AddRowVector(this DenseMatrix lhs, DenseVector rhs, DenseMatrix sum)
        {
            if (lhs.ColumnCount != rhs.Count)
                throw Exceptions.CreateInvalidVectorDimensionException(nameof(rhs), nameof(lhs.ColumnCount));

            if (!lhs.DimensionEqualsTo(sum))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(sum), nameof(lhs.RowCount), nameof(lhs.ColumnCount));

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var sumArray = sum.AsColumnMajorArray();
            var (numRows, numCols) = (sum.RowCount, sum.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var s = sumArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var rest = numRows % UNROLL_STEP_NUM;
                var end = numRows - rest;
                for (var i = 0; i < end; i += UNROLL_STEP_NUM)
                {
                    var lhsVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(l[i..]));
                    var sVec = lhsVec + rhsVec;
                    sVec.StoreUnsafe(ref MemoryMarshal.GetReference(s[i..]));
                }

                // 端数処理
                for(var i = end; i < numRows; i++)
                    s[i] = l[i] + rhs[j];
            });
        }

        /// <summary>
        /// 密行列に密列ベクトルをブロードキャストして減算する.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <param name="diff"></param>
        public static void SubtractColumnVector(this DenseMatrix lhs, DenseVector rhs, DenseMatrix diff)
        {
            if (lhs.RowCount != rhs.Count)
                throw Exceptions.CreateInvalidVectorDimensionException(nameof(rhs), nameof(lhs.RowCount));

            if (!lhs.DimensionEqualsTo(diff))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(diff), nameof(lhs.RowCount), nameof(lhs.ColumnCount));

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var rhsArray = rhs.AsArray();
            var sumArray = diff.AsColumnMajorArray();
            var (numRows, numCols) = (diff.RowCount, diff.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var d = sumArray.AsSpan(j * numRows);
                var rest = numRows % UNROLL_STEP_NUM;
                var end = numRows - rest;
                for (var i = 0; i < end; i += UNROLL_STEP_NUM)
                {
                    var leftVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(l[i..]));
                    var rightVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(rhsArray.AsSpan(i)));
                    var diffVec = leftVec - rightVec;
                    diffVec.StoreUnsafe(ref MemoryMarshal.GetReference(d[i..]));
                }

                // 端数処理
                for(var i = end; i < numRows; i++)
                    d[i] = l[i] - rhsArray[i];
            });
        }

        /// <summary>
        /// 密行列に密行ベクトルをブロードキャストして減算する.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <param name="diff"></param>
        public static void SubtractRowVector(this DenseMatrix lhs, DenseVector rhs, DenseMatrix diff)
        {
            if (lhs.ColumnCount != rhs.Count)
                throw Exceptions.CreateInvalidVectorDimensionException(nameof(rhs), nameof(lhs.ColumnCount));

            if (!lhs.DimensionEqualsTo(diff))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(diff), nameof(lhs.RowCount), nameof(lhs.ColumnCount));

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var diffArray = diff.AsColumnMajorArray();
            var (numRows, numCols) = (diff.RowCount, diff.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var d = diffArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var rest = numRows % UNROLL_STEP_NUM;
                var end = numRows - rest;
                for (var i = 0; i < end; i += UNROLL_STEP_NUM)
                {
                    var lhsVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(l[i..]));
                    var diffVec = lhsVec - rhsVec;
                    diffVec.StoreUnsafe(ref MemoryMarshal.GetReference(d[i..]));
                }

                // 端数処理
                for(var i = end; i < numRows; i++)
                    d[i] = l[i] - rhs[j];
            });
        }


        /// <summary>
        /// 密行列に密行ベクトルをブロードキャストして除算する.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <param name="sum"></param>
        public static void DivideByRowVector(this DenseMatrix lhs, DenseVector rhs, DenseMatrix quotient)
        {
            if (lhs.ColumnCount != rhs.Count)
                throw Exceptions.CreateInvalidVectorDimensionException(nameof(rhs), nameof(lhs.ColumnCount));

            if (!lhs.DimensionEqualsTo(quotient))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(quotient), nameof(lhs.RowCount), nameof(lhs.ColumnCount));

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var quotientArray = quotient.AsColumnMajorArray();
            var (numRows, numCols) = (quotient.RowCount, quotient.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var q = quotientArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var rest = numRows % UNROLL_STEP_NUM;
                var end = numRows - rest;
                for (var i = 0; i < end; i += UNROLL_STEP_NUM)
                {
                    var lhsVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(l[i..]));
                    var qVec = lhsVec / rhsVec;
                    qVec.StoreUnsafe(ref MemoryMarshal.GetReference(q[i..]));
                }

                // 端数処理
                for(var i = end; i < numRows; i++)
                    q[i] = l[i] / rhs[j];
            });
        }

        public static void PointwiseSigmoid(this DenseMatrix x, DenseMatrix y)
        {
            if(!x.DimensionEqualsTo(y))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(y), nameof(x.RowCount), nameof(y.RowCount));

            x.Negate(y);
            y.PointwiseExp(y);
            y.Add(1.0f, y);
            y.Map(x => 1.0f / x, y);
        }

        public static void ColumnSoftmax(this DenseMatrix x, DenseMatrix y)
        {
            if(!x.DimensionEqualsTo(y))
                throw Exceptions.CreateInvalidMatrixDimensionException(nameof(y), nameof(x.RowCount), nameof(y.RowCount));

            var colMax = x.ColumnMax();
            x.SubtractRowVector(colMax, y);
            y.PointwiseExp(y);
            var colSums = (DenseVector)y.ColumnSums();
            y.DivideByRowVector(colSums, y);
        }

        public static DenseVector ColumnMax(this DenseMatrix x)
        {
            var maxVec = DenseVector.Create(x.ColumnCount, 0.0f);
            var xArray = x.AsColumnMajorArray();
            Parallel.For(0, x.ColumnCount, j =>
            {
                var col = xArray.AsSpan(j * x.RowCount, x.RowCount);
                var max = float.NegativeInfinity;
                for(var i = 0; i < col.Length; i++)
                {
                    if(col[i] > max)
                        max = col[i];
                }
                maxVec[j] = max;
            });

            return maxVec;
        }

        public static bool DimensionEqualsTo<T>(this Matrix<T> lhs, Matrix<T> rhs) where T : struct, IEquatable<T>, IFormattable
            => lhs.RowCount == rhs.RowCount && rhs.ColumnCount == lhs.ColumnCount;

        internal static DenseMatrix CopyToOrClone(this DenseMatrix src, DenseMatrix? dest)
        {
            if (dest is null)
                return (DenseMatrix)src.Clone();

            if (!dest.DimensionEqualsTo(src))
                return (DenseMatrix)src.Clone();

            src.CopyTo(dest);

            return dest;
        }
    }
}
