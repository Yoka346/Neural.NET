using NeuralNET.Layers.Activation;

namespace NeuralNETTests.Layers.Activation;

[TestClass]
public class SigmoidLayerTests
{
    [TestMethod]
    public void BackwardTest() 
    {
        Utils.TestPointwiseBackward(new SigmoidLayer(saveOutputRef:true));
        Utils.TestPointwiseBackward(new SigmoidLayer(saveOutputRef:false));
    }
}