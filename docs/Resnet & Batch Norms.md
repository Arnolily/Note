# My comprehension of Resnet and Batch Norms
***Resnet and Batch Normalization has been without doubt two of the most important elements in Deep Learning. This diary is to record my own understanding on why they work.***

## So let's first talk about BN.
- Why Normalize?

As we all know, what models do is basicly approximating a complicated function. Mathematically:  $$Y = \mathcal{F}\{X\}$$
Where \(\mathcal{F}\) is the mapping function between input \(X\) and output \(Y\). It would be easier to find such function if the input and output dataset follows the same distribution, and such mapping function would also be simpler, and simpler alleviate overfit problems.

This is easy to understand: if your inputs follow distribution A and labels follow distribution B, then the mapping function would be uniquely A to B. This would cause generalization defect.

- Why not Standard Gaussian Distribution?

Yes, there is a linear transformation (ie. scale and shift) after feature map \(X\) is normalized.

![](BN.png){: style="height:300px;"}

Let us explain this with an example.

    net=nn.Sequential(
        nn.Conv2d(),
        nn.BatchNorm2d(),
        nn.Sigmoid()
    )

In this net, we use a sigmoid activation function.

![](sigmoid-function.png){: style="height:300px;"}

It is pretty clear that for \(x\in (-1, 1)\), we have \(\sigma(x)\approx x\) . Which means the activation function is similar to an identity transformation. That is not a good sign, because the purpose of activation function is to add non-linearity in to the model. Therefore, without scale and shift, the non-linearity will be weak and the fitting process would be hard.

ReLU, on the other hand, would not be heavily affected. So intuitionally speaking, it should be fine if we are using ReLU as activation function. But practically speaking, it seems better if we also let the model itself decide.
