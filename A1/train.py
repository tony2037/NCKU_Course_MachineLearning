import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # See https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

train_filename = './train.tfrecords'
validation_filename = './validation.tfrecords'

train_set_path = './training/'
valid_set_path = './validation/'

class Dataset():
    def __init__(self, train_path, validation_path):
        self.filesnames = []
        self.labels = []
        for i in range(10):
            images = glob('%sSample%s/*.png' % (train_path, str(i+1).zfill(3)))
            label = [0]* 10
            label[i] = 1
            for image in images:
                self.filesnames.append(image)
                self.labels.append(label)

        self.filesnames = tf.constant(self.filesnames)
        self.labels = tf.constant(self.labels)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filesnames, self.labels))
        self.dataset = self.dataset.map(self._parse_function)
        self.dataset = self.dataset.shuffle(buffer_size=1000).batch(32).repeat(1000)

        self.iterator = self.dataset.make_one_shot_iterator()
    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label


class CNN():
    def __init__(self, train_filename_queue, validation_filename_queue, dataset):
        self.train_filename_queue = train_filename_queue
        self.validation_filename_queue = validation_filename_queue
        self.dataset = dataset

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def model_summary(self):
        model_vars = tf.trainable_variables()
        # model_vars = tf.model_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def build(self):
        features, labels = self.dataset.iterator.get_next()
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        # define placeholder for inputs to network
        self.keep_prob = tf.placeholder(tf.float32)

        ## conv1 layer ##
        self.W_conv1 = self.weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
        self.b_conv1 = self.bias_variable([32])
        self.h_conv1 = tf.nn.relu(self.conv2d(features, self.W_conv1) + self.b_conv1) # output size 28x28x32
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)                                         # output size 14x14x32

        ## conv2 layer ##
        self.W_conv2 = self.weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
        self.b_conv2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) # output size 14x14x64
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)                                         # output size 7x7x64

        ## fc1 layer ##
        self.W_fc1 = self.weight_variable([7*7*64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, 0.5)

        ## fc2 layer ##
        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
        self.prediction = tf.clip_by_value(self.prediction, 1e-8, 1.0)
        print('The prediction shape: ')
        print(self.prediction.shape)
        print(self.prediction.dtype)
        print('The lables shape: ')
        print(labels.shape)
        print(labels.dtype)
        # the error between prediction and real data
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.prediction),
                                                    reduction_indices=[1]))       # loss
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # The model information
        self.model_summary()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(500):
                _, loss_value = sess.run([self.train_step, self.cross_entropy])
                print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

    def train(self,epochs):
        sess = tf.Session()
        # important step
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(epochs):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(self.train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.5})
            if i % 50 == 0:
                print(compute_accuracy(
                    mnist.test.images[:1000], mnist.test.labels[:1000]))

    def decode_from_tfrecords(self, filename_queue, is_batch):
        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   # return file and file_name, do not care about name
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                        })  #parse feature
        image = tf.decode_raw(features['img_raw'],tf.float64)
        image = tf.reshape(image, [56,56])
        label = tf.cast(features['label'], tf.float64)
        
        if is_batch:
            batch_size = 3
            min_after_dequeue = 10
            capacity = min_after_dequeue+3*batch_size
            image, label = tf.train.shuffle_batch([image, label],
                                                            batch_size=batch_size, 
                                                            num_threads=3, 
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        return image, label

if __name__ == '__main__':

    # Construct a Dataset object
    Dataset = Dataset(train_set_path, valid_set_path)

    train_filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None) # read in the stream
    validation_filename_queue = tf.train.string_input_producer([validation_filename],num_epochs=None) # read in the stream
    
    print('Build up model')
    model = CNN(train_filename_queue = train_filename_queue, validation_filename_queue = validation_filename_queue, dataset = Dataset)
    model.build()

    print('Data proccess')
    # run_test = True
    filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None) # read in the stream
    train_image, train_label = model.decode_from_tfrecords(filename_queue, is_batch=True)
    print(train_image.shape)

    filename_queue = tf.train.string_input_producer([validation_filename],num_epochs=None) # read in the stream
    valid_image, valid_label = model.decode_from_tfrecords(filename_queue, is_batch=True)

