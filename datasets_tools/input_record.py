#-*-coding:utf-8-*-
'''
从record_file中读取图像或者文件。
shuffle为是否随机，对于train，一般为true，对eval和predict，为false
IMG_HEIGHT,IMG_WIDTH,IMG_CHANNEL是图像的三个信息，宽高，通道号
batch_size为批量读取数据
num_epochs对应训练的epochs，即加载的次数。对eval和predict来说，一般为1。对于训练，一般为大于1
'''
import tensorflow as tf

IMG_HEIGHT = 1080
IMG_WIDTH= 1920
IMG_CHANNEL = 3

def _parse_function(record):
    """
    Extract data from a `tf.Example` protocol buffer.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'label/points': tf.FixedLenFeature([136], tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)

    # Extract features from single example
    image_decoded = tf.image.decode_image(parsed_features['image/encoded'])
    image_reshaped = tf.reshape(
        image_decoded, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    points = tf.cast(parsed_features['label/points'], tf.float32)

    return {"x": image_reshaped, "name": parsed_features['image/filename']}, points


def input_fn(record_file, batch_size, num_epochs=None, shuffle=True):
    """
    Input function required for TensorFlow Estimator.
    """
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    feature, label = iterator.get_next()
    return feature, label

def serving_input_receiver_fn(): #创建出相关的空间，用来存储
    """An input receiver that expects a serialized tf.Example."""
    image = tf.placeholder(dtype=tf.uint8,
                           shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
                           name='input_image_tensor')
    receiver_tensor = {'image': image}
    feature = tf.reshape(image, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    return tf.estimator.export.ServingInputReceiver(feature, receiver_tensor)