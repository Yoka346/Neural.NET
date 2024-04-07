using NeuralNET.Layers.Activation;

namespace NeuralNETTests.Layers.Activation;

[TestClass]
public class SoftmaxLayerTests
{
    [TestMethod]
    public void BackwardTest() 
    {
        Utils.TestBackward(new SoftmaxLayer(saveOutputRef:true));
        Utils.TestBackward(new SoftmaxLayer(saveOutputRef:false));
    }
}