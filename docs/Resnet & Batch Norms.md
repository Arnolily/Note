# __My comprehension of Resnet and Batch Norms__
***Resnet and Batch Normalization has been without doubt two of the most important elements in Deep Learning. This diary is to record my own understanding on why they work.***

## __Let's first talk about BN.__
### **Improving Numerical stability**

Suppose we have a 1-layer Linear Neural Network for regression problem, the output \(O\) and MSE loss \(\mathcal{L}_{MSE}\) is:
$$O=\sigma(W \cdot X)$$ $$\mathcal{L}_{MSE}=\frac{1} {2}(O-Y)^2$$
Through back propagation, \(\nabla \mathcal{L}_{W}=(O-Y)\cdot \frac{\partial O} {\partial (W)}=O\cdot X\cdot \sigma'\)  
If the net has N layers, then the gradient of the first layer would be \(O \cdot \prod_{i=1}^{N} \frac{\partial O} {\partial (W)}\), and the product of this sequence would tend to 0 or _inf_. This is __Numerical instability__.

With BN, we could restric X into any Gaussian distribution, thus fixing the instability of \(X\) and \(\sigma'\).

### **(Perhaps not) Alleviate Internal Covariate Shift** [1]
In the original paper of Batch Normalizaion, they defined ICS.


>ICS is the phenomenon where the distribution of inputs to a layer in the network changes due to an update of parameters of the previous layers.


Mathematically, each layer can be derived as:  $$Y = \mathcal{F}\{X\}$$
Where \(\mathcal{F}\) is the mapping function between input \(X\) and output \(Y\). Basically, they are learning this map between input pattern and output pattern. However, due to ICS, the input pattern could be constantly changing, and such changes could break the learnt mapping function; therefore, ICS is believed to have detrimental effect on training process.

However, other researchers had shown that ICS will not effect performance and BN does not reduce ICS. They prefer to explain the effectiveness of BN in another way.

### **Smoother Loss landscape** [2]
These researchers shown that BN contributes to the smoother Loss landscape, which could be visualized:
![](Landscape.png)

With BN, the Loss landscape is smoother, training efficiency will be improved, and that is why higher learning rate could be utilized. 

### **Better use of non-linearity** [3]

Please be noted, there is a linear transformation (ie. scale and shift) after feature map \(x\) is normalized.


![](BN.png){: style="height:300px;"}


Let us suppose we use sigmoid function as activation function.

![](sigmoid-function.png){: style="height:300px;"}

It is pretty clear that for \(x\in (-1, 1)\), we have \(\sigma(x)\approx x\) . Which means the activation function is similar to an identity transformation. That is not a good sign, because the purpose of activation function is to add non-linearity in to the model. Therefore, without scale and shift, the non-linearity will be weak and the fitting process would be hard.

Even for ReLU, which seems not going to be heavily effected, also witnessed the situation where all the neurons are either activated \(x>0\) or deactivated \(x<0\). Be aware that if a ReLU neuron is always active, then it is linear; if it is always deactive, then it does not exist.

But in my perspective, this might not be the most important contribution of BN. Experiments have proved that if we put BN after activation, network could even sometimes outperform!

## Citation
[1] Batch normalization: Accelerating deep network training by reducing internal covariate shift.
[2] How Does Batch Normalization Help Optimization?  
[3] The shattered gradients problem: If resnets are the answer, then what is the question?