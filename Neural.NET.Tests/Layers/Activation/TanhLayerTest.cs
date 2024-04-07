using NeuralNET.Layers.Activation;

namespace NeuralNETTests.Layers.Activation;

[TestClass]
public class TanhLayerTests
{
    [TestMethod]
    public void BackwardTest() 
    {
        Utils.TestBackward(new TanhLayer(saveOutputRef:true));
        Utils.TestBackward(new TanhLayer(saveOutputRef:false));
    }
}