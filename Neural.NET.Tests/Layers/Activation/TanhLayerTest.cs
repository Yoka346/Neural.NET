using NeuralNET.Layers.Activation;

namespace NeuralNETTests.Layers.Activation;

[TestClass]
public class TanhLayerTests
{
    [TestMethod]
    public void BackwardTest() 
    {
        Utils.TestPointwiseBackward(new TanhLayer(saveOutputRef:true));
        Utils.TestPointwiseBackward(new TanhLayer(saveOutputRef:false));
    }
}