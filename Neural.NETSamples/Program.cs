using System;

using MathNet.Numerics.Providers.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

using Neural.NET.Layers.Arithmetic;

// Intel Math Kernal Library(MKL)を有効化する.
if (LinearAlgebraControl.TryUseNativeMKL())
    Console.WriteLine("Use Intel MKL");

var apple = DenseMatrix.Create(1, 1, 100.0f);
var appleNum = DenseMatrix.Create(1, 1, 2.0f);
var orange = DenseMatrix.Create(1, 1, 150.0f);
var orangeNum = DenseMatrix.Create(1, 1, 3.0f); 
var tax = DenseMatrix.Create(1, 1, 1.1f);

// layer
var mulAppleLayer = new MultiplyLayer(saveInputRef:true);
var mulOrangelayer = new MultiplyLayer(saveInputRef: true);
var addAppleOrangeLayer = new AddLayer();
var mulTaxLayer = new MultiplyLayer(saveInputRef: true);

// forward
var applePrice = mulAppleLayer.Forward(apple, appleNum);
var orangePrice = mulOrangelayer.Forward(orange, orangeNum);
var allPrice = addAppleOrangeLayer.Forward(applePrice, orangePrice);
var price = mulTaxLayer.Forward(allPrice, tax);

// backward
var dPrice = DenseMatrix.Create(1, 1, 1.0f);
(var dAllPrice, var dTax) = mulTaxLayer.Backward(dPrice);
(var dApplePrice, var dOrangePrice) = addAppleOrangeLayer.Backward(dAllPrice);
(var dOrange, var dOrangeNum) = mulOrangelayer.Backward(dOrangePrice);
(var dApple, var dAppleNum) = mulAppleLayer.Backward(dApplePrice);

Console.WriteLine(price[0, 0]);
Console.WriteLine($"{dAppleNum[0, 0]} {dApple[0, 0]} {dOrange[0, 0]} {dOrangeNum[0, 0]} {dTax[0, 0]}");