using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

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
            {
                var message = $"Vector dimension must agree: {nameof(rhs)} is {lhs.RowCount}.";
                throw new ArgumentException(message, nameof(rhs));
            }

            if (lhs.RowCount != sum.RowCount || lhs.ColumnCount != sum.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(sum)} is {lhs.RowCount}x{lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(sum));
            }

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var rhsArray = rhs.AsArray();
            var sumArray = sum.AsColumnMajorArray();
            var (numRows, numCols) = (sum.RowCount, sum.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var s = sumArray.AsSpan(j * numRows);
                var end = (numRows % UNROLL_STEP_NUM == 0) ? numRows : numRows - UNROLL_STEP_NUM;
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
            {
                var message = $"Vector dimension must agree: {nameof(rhs)} is {lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(rhs));
            }

            if (lhs.RowCount != sum.RowCount || lhs.ColumnCount != sum.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(sum)} is {lhs.RowCount}x{lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(sum));
            }

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var sumArray = sum.AsColumnMajorArray();
            var (numRows, numCols) = (sum.RowCount, sum.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var s = sumArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var end = (numRows % UNROLL_STEP_NUM == 0) ? numRows : numRows - UNROLL_STEP_NUM;
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
            {
                var message = $"Vector dimension must agree: {nameof(rhs)} is {lhs.RowCount}.";
                throw new ArgumentException(message, nameof(rhs));
            }

            if (lhs.RowCount != diff.RowCount || lhs.ColumnCount != diff.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(diff)} is {lhs.RowCount}x{lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(diff));
            }

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var rhsArray = rhs.AsArray();
            var sumArray = diff.AsColumnMajorArray();
            var (numRows, numCols) = (diff.RowCount, diff.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var d = sumArray.AsSpan(j * numRows);
                var end = (numRows % UNROLL_STEP_NUM == 0) ? numRows : numRows - UNROLL_STEP_NUM;
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
            {
                var message = $"Vector dimension must agree: {nameof(rhs)} is {lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(rhs));
            }

            if (lhs.RowCount != diff.RowCount || lhs.ColumnCount != diff.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(diff)} is {lhs.RowCount}x{lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(diff));
            }

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var diffArray = diff.AsColumnMajorArray();
            var (numRows, numCols) = (diff.RowCount, diff.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var d = diffArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var end = (numRows % UNROLL_STEP_NUM == 0) ? numRows : numRows - UNROLL_STEP_NUM;
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
            {
                var message = $"Vector dimension must agree: {nameof(rhs)} is {lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(rhs));
            }

            if (lhs.RowCount != quotient.RowCount || lhs.ColumnCount != quotient.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(quotient)} is {lhs.RowCount}x{lhs.ColumnCount}.";
                throw new ArgumentException(message, nameof(quotient));
            }

            const int UNROLL_STEP_NUM = 8;
            var lhsArray = lhs.AsColumnMajorArray();
            var quotientArray = quotient.AsColumnMajorArray();
            var (numRows, numCols) = (quotient.RowCount, quotient.ColumnCount);
            Parallel.For (0, numCols, j =>
            {
                var l = lhsArray.AsSpan(j * numRows);
                var q = quotientArray.AsSpan(j * numRows);
                var rhsVec = Vector256.Create(rhs[j]);
                var end = (numRows % UNROLL_STEP_NUM == 0) ? numRows : numRows - UNROLL_STEP_NUM;
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
            x.Negate(y);
            y.PointwiseExp(y);
            y.Add(1.0f, y);
            y.Map(x => 1.0f / x, y);
        }

        public static void ColumnSoftmax(this DenseMatrix x, DenseMatrix y)
        {
            x.PointwiseExp(y);
            var colSums = (DenseVector)y.ColumnSums();
            y.DivideByRowVector(colSums, y);
        }

        internal static DenseMatrix CopyToOrClone(this DenseMatrix src, DenseMatrix? dest)
        {
            if (dest is null)
                return (DenseMatrix)src.Clone();

            if (dest.RowCount != src.RowCount || dest.ColumnCount != src.ColumnCount)
                return (DenseMatrix)src.Clone();

            src.CopyTo(dest);

            return dest;
        }
    }
}
