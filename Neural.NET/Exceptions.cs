using System;

namespace NeuralNET
{
    public static class Exceptions
    {
        public static Exception CreateBackwardBeforeForwardException() => new InvalidOperationException("Backward method must be called after forward.");

        public static Exception CreateInvalidVectorDimensionException(string value, string dim) 
            => new ArgumentException(value, $"Vector dimension must agree: {value} is {dim}.");

        public static Exception CreateInvalidMatrixDimensionException(string value, string rowCount, string colCount)
            => new ArgumentException(value, $"Matrix dimensions must agree: {value} is {rowCount}x{colCount}.");
    }
}