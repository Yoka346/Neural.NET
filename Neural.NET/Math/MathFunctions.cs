using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNET.Math
{
    public static class MathFunctions
    {
        public static float StdSigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));
    }
}
