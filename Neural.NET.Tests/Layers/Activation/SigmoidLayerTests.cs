using NeuralNET.Layers.Activation;

namespace NeuralNETTests.Layers.Activation;

[TestClass]
public class SigmoidLayerTests
{
    [TestMethod]
    public void BackwardTest() 
    {
        Utils.TestBackward(new SigmoidLayer(saveOutputRef:true));
        Utils.TestBackward(new SigmoidLayer(saveOutputRef:false));
    }
}