
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

训练

1.读入数据，将下面代码写入models/slim/datasets/dataset_classification.py。

    import os
    import tensorflow as tf
    slim = tf.contrib.slim

    def get_dataset(dataset_dir, num_samples, num_classes, labels_to_names_path=None, file_pattern='*.tfrecord'):
        file_pattern = os.path.join(dataset_dir, file_pattern)
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        items_to_descriptions = {
            'image': 'A color image of varying size.',
            'label': 'A single integer between 0 and ' + str(num_classes - 1),
        }
        labels_to_names = None
        if labels_to_names_path is not None:
            fd = open(labels_to_names_path)
            labels_to_names = {i : line.strip() for i, line in enumerate(fd)}
            fd.close()
        return slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=num_samples,
                items_to_descriptions=items_to_descriptions,
                num_classes=num_classes,
                labels_to_names=labels_to_names)
                
2.构建模型

官方提供了许多模型在models/slim/nets/

3.开始训练

官方提供的训练脚本为

    CUDA_VISIBLE_DEVICES="0" python train_image_classifier.py \
        --train_dir=train_logs \
        --dataset_name=flowers \
        --dataset_split_name=train \
        --dataset_dir=../../data/flowers \
        --model_name=inception_resnet_v2 \
        --checkpoint_path=../../checkpoints/inception_resnet_v2_2016_08_30.ckpt \
        --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --max_number_of_steps=1000 \
        --batch_size=32 \
        --learning_rate=0.01 \
        --learning_rate_decay_type=fixed \
        --save_interval_secs=60 \
        --save_summaries_secs=60 \
        --log_every_n_steps=10 \
        --optimizer=rmsprop \
        --weight_decay=0.00004
        
不fine-tune把--checkpoint_path, --checkpoint_exclude_scopes和--trainable_scopes删掉。

fine-tune所有层把--checkpoint_exclude_scopes和--trainable_scopes删掉。

如果只使用CPU则加上--clone_on_cpu=True。

其它参数可删掉用默认值或自行修改。

使用自己的数据则需要修改models/slim/train_image_classifier.py

    把
    from datasets import dataset_factory
    修改为
    from datasets import dataset_classification
    
    把
    dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    修改为
    dataset = dataset_classification.get_dataset(
    FLAGS.dataset_dir, FLAGS.num_samples, FLAGS.num_classes, FLAGS.labels_to_names_path)
    
    在
    tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
    后加入
    tf.app.flags.DEFINE_integer(
    'num_samples', 3320, 'Number of samples.')

    tf.app.flags.DEFINE_integer(
        'num_classes', 5, 'Number of classes.')

    tf.app.flags.DEFINE_string(
        'labels_to_names_path', None, 'Label names file path.')
      
训练时执行以下命令：

    python train_image_classifier.py \
    --train_dir=train_logs \
    --dataset_dir=../../data/train \
    --num_samples=3320 \
    --num_classes=5 \
    --labels_to_names_path=../../data/labels.txt \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=../../checkpoints/inception_resnet_v2_2016_08_30.ckpt \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits
    
4.可视化log

    tensorboard --logdir train_logs/
    
验证

官方提供了验证脚本

    python eval_image_classifier.py \
        --checkpoint_path=train_logs \
        --eval_dir=eval_logs \
        --dataset_name=flowers \
        --dataset_split_name=validation \
        --dataset_dir=../../data/flowers \
        --model_name=inception_resnet_v2
        
同理，如果是自己的数据集，则需要修改models/slim/eval_image_classifier.py

                   把
                   from datasets import dataset_factory
                   修改为
                   from datasets import dataset_classification

                   把
                   dataset = dataset_factory.get_dataset(
                    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
                   修改为
                   dataset = dataset_classification.get_dataset(
                    FLAGS.dataset_dir, FLAGS.num_samples, FLAGS.num_classes, FLAGS.labels_to_names_path)

    在
    tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
    后加入
    tf.app.flags.DEFINE_integer(
    'num_samples', 350, 'Number of samples.')

    tf.app.flags.DEFINE_integer(
        'num_classes', 5, 'Number of classes.')

    tf.app.flags.DEFINE_string(
        'labels_to_names_path', None, 'Label names file path.')
        
验证时执行以下命令：

    python eval_image_classifier.py \
        --checkpoint_path=train_logs \
        --eval_dir=eval_logs \
        --dataset_dir=../../data/val \
        --num_samples=350 \
        --num_classes=5 \
        --model_name=inception_resnet_v2
        
可视化log命令为

    tensorboard --logdir eval_logs/ --port 6007





