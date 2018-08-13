本文主要讲解Inception结构。

Inception v1网络

![image](https://github.com/15458wew/slim/blob/master/images/inception.png)

如上图所示，实验中Input层大小为28*28*192，（黑色方框中的内容为改进部分），则

未改进前（去掉黑色方框部分）weights大小为：1*1*192*64+3*3*192*128+5*5*192*32=387072

未改进前feature map大小为：28*28*64+28*28*128+28*28*32+28*28*192=28*28*416

改进后weights大小为：1*1*192*64+(1*1*192*96+3*3*96*128)+(1*1*192*16+5*5*16*32)+1*1*192*32=163328

改进后feature map大小为：28*28*64+28*28*128+28*28*32+28*28*32=28*28*256

经过改良，weights和dimension都减少了。

同时使用了1*1，3*3，5*5的卷积，增加了网络对尺寸的适应性。

Inception v1结构有3个输出（类比FCN的8s，16s，32s），最后一个softmax前使用的是global average pooling
