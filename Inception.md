本文主要讲解Inception结构。

Inception v1网络

![image](https://github.com/15458wew/slim/blob/master/images/inception.png)

如上图所示，实验中Input层大小为28*28*192，（黑色方框中的内容为改进部分），则

未改进前（去掉黑色方框部分）weights大小为：1*1*192*64+3*3*192*128+5*5*192*32=387072

未改进前feature map大小为：28*28*64+28*28*128+28*28*32+28*28*192=28*28*416

