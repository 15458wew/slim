
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
        
 4.生成TF-record数据（TF-record数据为二进制流格式，可加速tensorflow的训练）
 
    import sys
    sys.path.insert(0, '../models/slim/')
    from datasets import dataset_utils
    import math
    import os
    import tensorflow as tf

    def convert_dataset(list_path, data_dir, output_dir, _NUM_SHARDS=5):
        fd = open(list_path)
        lines = [line.split() for line in fd]
        fd.close()
        num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
        with tf.Graph().as_default():
            decode_jpeg_data = tf.placeholder(dtype=tf.string)
            decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
            with tf.Session('') as sess:
                for shard_id in range(_NUM_SHARDS):
                    output_path = os.path.join(output_dir,
                        'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
                    tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                            i + 1, len(lines), shard_id))
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
                        image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                        height, width = image.shape[0], image.shape[1]
                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, int(lines[i][1]))
                        tfrecord_writer.write(example.SerializeToString())
                    tfrecord_writer.close()
        sys.stdout.write('\n')
        sys.stdout.flush()

    os.system('mkdir -p train')
    convert_dataset('list_train.txt', 'flower_photos', 'train/')
    os.system('mkdir -p val')
    convert_dataset('list_val.txt', 'flower_photos', 'val/')
    
5.下载模型（可选，这里以inception-resnet-v2为例）

    wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    tar zxf inception_resnet_v2_2016_08_30.tar.gz

