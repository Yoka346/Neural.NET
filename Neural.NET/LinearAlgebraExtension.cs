using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuralNET
{
    public static class LinearAlgebraExtension
    {

        /// <summary>
        /// 密行列に密列ベクトルをブロードキャストして加算する.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="sum"></param>
        public static void AddColumnVector(this DenseMatrix left, DenseVector right, DenseMatrix sum)
        {
            if (left.RowCount != right.Count)
            {
                var message = $"Vector dimension must agree: {nameof(right)} is {left.RowCount}.";
                throw new ArgumentException(message, nameof(right));
            }

            if (left.RowCount != sum.RowCount || left.ColumnCount != sum.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(sum)} is {left.RowCount}x{left.ColumnCount}.";
                throw new ArgumentException(message, nameof(sum));
            }

            var leftArray = left.AsColumnMajorArray();
            var rightArray = right.AsArray();
            var sumArray = sum.AsColumnMajorArray();
            var (rowNum, colNum) = (sum.RowCount, sum.ColumnCount);

            unsafe
            {
                const int UNROLL_STEP_NUM = 8;
                fixed (float* leftPtr = leftArray)
                fixed (float* rightPtr = rightArray)
                fixed (float* sumPtr = sumArray)
                {
                    for (var j = 0; j < colNum; j++)
                    {
                        var l = leftPtr + j * rowNum;
                        var s = sumPtr + j * rowNum;
                        for (var i = 0; i < rowNum; i += UNROLL_STEP_NUM)
                        {
                            var leftVec = Avx.LoadVector256(l + i);
                            var rightVec = Avx.LoadVector256(rightPtr + i);
                            var sumVec = Avx.Add(leftVec, rightVec);
                            Avx.Store(s + i, sumVec);
                        }
                    }

                    // 端数処理
                    for (var j = 0; j < colNum; j++)
                    {
                        var l = leftPtr + j * rowNum;
                        var s = sumPtr + j * rowNum;
                        for (var i = UNROLL_STEP_NUM * (rowNum / UNROLL_STEP_NUM); i < rowNum; i++)
                            s[i] = l[i] + rightPtr[i];
                    }
                }
            }
        }

        /// <summary>
        /// 密行列に密行ベクトルをブロードキャストして加算する.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="sum"></param>
        public static void AddRowVector(this DenseMatrix left, DenseVector right, DenseMatrix sum)
        {
            if (left.RowCount != right.Count)
            {
                var message = $"Vector dimension must agree: {nameof(right)} is {left.ColumnCount}.";
                throw new ArgumentException(message, nameof(right));
            }

            if (left.RowCount != sum.RowCount || left.ColumnCount != sum.ColumnCount)
            {
                var message = $"Matrix dimensions must agree: {nameof(sum)} is {left.RowCount}x{left.ColumnCount}.";
                throw new ArgumentException(message, nameof(sum));
            }

            var leftArray = left.AsColumnMajorArray();
            var rightArray = right.AsArray();
            var sumArray = sum.AsColumnMajorArray();
            var (rowNum, colNum) = (sum.RowCount, sum.ColumnCount);

            unsafe
            {
                const int UNROLL_STEP_NUM = 8;
                fixed (float* leftPtr = leftArray)
                fixed (float* rightPtr = rightArray)
                fixed (float* sumPtr = sumArray)
                {
                    for (var j = 0; j < colNum; j++)
                    {
                        var n = rightPtr[j];
                        var offset = j * rowNum;
                        var l = leftPtr + offset;
                        var s = sumPtr + offset;
                        for (var i = 0; i < rowNum; i += UNROLL_STEP_NUM)
                        {
                            var leftVec = Avx.LoadVector256(l + i);
                            var nVec = Avx.BroadcastScalarToVector256(&n);
                            var sumVec = Avx.Add(leftVec, nVec);
                            Avx.Store(s + i, sumVec);
                        }
                    }

                    for (var j = 0; j < colNum; j++)
                    {
                        var n = rightPtr[j];
                        var offset = j * rowNum;
                        var l = leftPtr + offset;
                        var s = sumPtr + offset;
                        for (var i = UNROLL_STEP_NUM * (rowNum / UNROLL_STEP_NUM); i < rowNum; i++)
                            s[i] = l[i] + n;
                    }
                }
            }
        }

        /// <summary>
        /// 密行列の各行の総和を計算する.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="sums"></param>
        /// <exception cref="ArgumentException"></exception>
        public static void RowSums(this DenseMatrix matrix, DenseVector sums)
        {
            if (matrix.RowCount != sums.Count)
            {
                var message = $"Vector dimension must agree: {nameof(sums)} is {matrix.RowCount}.";
                throw new ArgumentException(message, nameof(sums));
            }

            var matrixArray = matrix.AsColumnMajorArray();
            var sumsArray = sums.AsArray();
            var (rowNum, colNum) = (matrix.RowCount, matrix.ColumnCount);

            unsafe
            {
                const int UNROLL_STEP_NUM = 8;
                fixed (float* matrixPtr = matrixArray)
                fixed (float* sumPtr = sumsArray)
                {
                    for (var j = 0; j < colNum; j++)
                    {
                        var m = matrixPtr + j * rowNum;
                        for (var i = 0; i < rowNum; i += UNROLL_STEP_NUM)
                        {
                            var sVec = Avx.LoadVector256(sumPtr + i);
                            var mVec = Avx.LoadVector256(m + i);
                            sVec = Avx.Add(sVec, mVec);
                            Avx.Store(sumPtr + i, sVec);
                        }
                    }

                    // 端数処理
                    for (var j = 0; j < colNum; j++)
                    {
                        var m = matrixPtr + j * rowNum;
                        for (var i = UNROLL_STEP_NUM * (rowNum / UNROLL_STEP_NUM); i < rowNum; i++)
                            sumPtr[i] += m[i];
                    }
                }
            }
        }

        /// <summary>
        /// 密行列の各列の総和を計算する.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="sums"></param>
        /// <exception cref="ArgumentException"></exception>
        public static void ColumnSums(this DenseMatrix matrix, DenseVector sums)
        {
            if (matrix.ColumnCount != sums.Count)
            {
                var message = $"Vector dimension must agree: {nameof(sums)} is {matrix.ColumnCount}.";
                throw new ArgumentException(message, nameof(sums));
            }

            var matrixArray = matrix.AsColumnMajorArray();
            var sumsArray = sums.AsArray();
            var (rowNum, colNum) = (matrix.RowCount, matrix.ColumnCount);

            unsafe
            {
                const int UNROLL_STEP_NUM = 8;
                fixed (float* matrixPtr = matrixArray)
                fixed (float* sumPtr = sumsArray)
                {
                    for (var i = 0; i < rowNum; i += UNROLL_STEP_NUM)
                        for (var j = 0; j < colNum; j++)
                        {
                            var m = matrixPtr + i + j * rowNum;
                            sumPtr[j] += m[0] + m[1] + m[2] + m[3] + m[4] + m[5] + m[6] + m[7];
                        }

                    // 端数処理
                    for (var i = UNROLL_STEP_NUM * (rowNum / UNROLL_STEP_NUM); i < rowNum; i++)
                        for (var j = 0; j < colNum; j++)
                            sumPtr[j] += matrixPtr[i + j * rowNum];
                }
            }
        }

        public static void PointwiseSigmoid(this DenseMatrix x, DenseMatrix y)
        {
            x.Negate(y);
            y.PointwiseExp(y);
            y.Add(1.0f, y);
            y.Divide(1.0f, y);
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
