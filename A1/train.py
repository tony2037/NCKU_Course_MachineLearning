import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # See https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

train_filename = './train.tfrecords'
validation_filename = './validation.tfrecords'

class CNN():
    def __init__(self, train_filename_queue, validation_filename_queue):
        self.train_filename_queue = train_filename_queue
        self.validation_filename_queue = validation_filename_queue
        self.model_file_name = 'model.ckpt'

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
        # define placeholder for inputs to network
        self.xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
        self.ys = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(self.xs, [-1, 28, 28, 1])
        # print(x_image.shape)  # [n_samples, 28,28,1]

        ## conv1 layer ##
        W_conv1 = self.weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        h_pool1 = self.max_pool_2x2(h_conv1)                                         # output size 14x14x32

        ## conv2 layer ##
        W_conv2 = self.weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
        h_pool2 = self.max_pool_2x2(h_conv2)                                         # output size 7x7x64

        ## fc1 layer ##
        W_fc1 = self.weight_variable([7*7*64, 1024])
        b_fc1 = self.bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        ## fc2 layer ##
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        self.prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        print('The prediction shape: ')
        print(self.prediction.shape)

        # the error between prediction and real data
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction),
                                                    reduction_indices=[1]))       # loss
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # The model information
        self.model_summary()

        # Build up Saver to save the model
        self.saver = tf.train.Saver()

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

        # Restore the model if exists
        if os.path.exists(self.model_file_name):
            self.saver.restore(sess, './%s' % self.model_file_name)

        for i in range(epochs):
            print('epochs : %s' %(str(i)))

            batch_xs, batch_ys = self.decode_from_tfrecords(self.train_filename_queue, is_batch = True)
            sess.run(self.train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.5})
            """
            if i % 50 == 0:
                print(compute_accuracy(
                    mnist.test.images[:1000], mnist.test.labels[:1000]))
            """
        # Save the model
        save_path = self.saver.save(sess, './%s' % self.model_file_name)
        print("Model saved in path: %s" % save_path)

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
            batch_size = 10 # Set batch_size
            min_after_dequeue = 10
            capacity = min_after_dequeue+3*batch_size
            image, label = tf.train.shuffle_batch([image, label],
                                                            batch_size=batch_size, 
                                                            num_threads=3, 
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        return image, label

if __name__ == '__main__':
    train_filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None) # read in the stream
    validation_filename_queue = tf.train.string_input_producer([validation_filename],num_epochs=None) # read in the stream
    
    print('Build up model')
    model = CNN(train_filename_queue = train_filename_queue, validation_filename_queue = validation_filename_queue)
    model.build()

    # Start training
    model.train(epochs = 1000)