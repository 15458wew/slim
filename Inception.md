 本文主要讲解Inception结构。

Inception v1网络

![image](https://github.com/15458wew/slim/blob/master/images/inception.png)

如上图所示，实验中Input层大小为28* 28* 192，（黑色方框中的内容为改进部分），则

未改进前（去掉黑色方框部分）weights大小为：1* 1* 192* 64 + 3* 3* 192* 128 + 5* 5* 192* 32=387072

未改进前feature map大小为：28* 28* 64 + 28* 28* 128 + 28* 28* 32 + 28* 28* 192=28*28*416

改进后weights大小为：1* 1* 192* 64 + (1* 1* 192* 96 + 3* 3* 96* 128)+(1* 1* 192* 16 + 5* 5* 16* 32) + 1* 1* 192* 32=163328

改进后feature map大小为：28* 28* 64 + 28* 28* 128 + 28* 28* 32 + 28* 28* 32 = 28* 28* 256

经过改良，weights和dimension都减少了。

同时使用了1*1，3*3，5*5的卷积，增加了网络对尺寸的适应性。

Inception v1结构有3个输出（类比FCN的8s，16s，32s），最后一个softmax前使用的是global average pooling


![image](https://github.com/15458wew/slim/blob/master/images/inception2.png)

Inception v2网络，Inception v3网络

在输入端加入BN层，个人理解为归一化，输出为标准正态分布N（0，1）


v2网络将一个5*5卷积分解为两个连续的3*3卷积，v3网络将3*3卷积分解为1*3卷积与3*1卷积。总体而言都是用更小的卷积核来替代大卷积核，增加网络深度，增加非线性

Inception v4网络，Inception-ResNet系列网络

该网络的改进之处就是使用上文的inception结构来替换resnet shortcut中的conv+1*1conv(resnet中用到的结构多为3*3，3*3，或者1*1，3*3，1*1。在这里与resnet用到的结构有不同之处）

![image](https://github.com/15458wew/slim/blob/master/images/resnet2.png)

resnet网络

个人理解，resnet网络想要克服深层网络准确率下降的问题（例如梯度消失），如上图所示，它将输入时的特征记为x，输出的特征记为y，y-x即为学习到的东西（残差），这里残差用F（x）表示。


![image](https://github.com/15458wew/slim/blob/master/images/resnet.png)

等号右边第一项表示损失函数的梯度，小括号中1表明短路机制可以无损传播（可以理解为深层网络复制浅层网络），另外一项代表经过有卷积的层，残差梯度不会全为-1。所以可以解决梯度消失的问题。
