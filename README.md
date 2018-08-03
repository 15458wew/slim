# slim

由于校园网网络状况不佳，代码上传存在问题。整体代码请参考 https://github.com/tensorflow/models/tree/master/research/slim

使用tensorflow-slim进行图像分类

准备

1.下载TF-slim图像模型库

    git clone https://github.com/tensorflow/models/

2.准备数据集，通过生成list.txt来表示图片路径与标签的关系

    import os

    class_names_to_ids = {'y0': 0, 'y1': 1, 'y2': 2, 'y3': 3, 'y4': 4}
    data_dir = 'datas/'
    output_path = 'list.txt'

    fd = open(output_path, 'w')
    for class_name in class_names_to_ids.keys():
        images_list = os.listdir(data_dir + class_name)
        for image_name in images_list:
            fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))

    fd.close()
    
3.随机生成数据集与验证集：

    import random

    _NUM_VALIDATION = 350
    _RANDOM_SEED = 0
    list_path = 'list.txt'
    train_list_path = 'list_train.txt'
    val_list_path = 'list_val.txt'

    fd = open(list_path)
    lines = fd.readlines()
    fd.close()
    random.seed(_RANDOM_SEED)
    random.shuffle(lines)

    fd = open(train_list_path, 'w')
    for line in lines[_NUM_VALIDATION:]:
        fd.write(line)

    fd.close()
    fd = open(val_list_path, 'w')
    for line in lines[:_NUM_VALIDATION]:
        fd.write(line)
