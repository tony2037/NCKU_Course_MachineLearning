import tensorflow as tf

train_filename = './train.tfrecords'
validation_filename = './validation.tfrecords'

class CNN():
    def __init__():
        pass
        
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def decode_from_tfrecords(filename_queue, is_batch):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # return file and file_name, do not care about name
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float64),
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
    # run_test = True
    filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None) # read in the stream
    train_image, train_label = decode_from_tfrecords(filename_queue, is_batch=True)

    filename_queue = tf.train.string_input_producer([validation_filename],num_epochs=None) # read in the stream
    valid_image, valid_label = decode_from_tfrecords(filename_queue, is_batch=True)
    with tf.Session() as sess: # Start a session
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        
        try:
            # while not coord.should_stop():
            for i in range(2):
                example, l = sess.run([train_image,train_label])# get image and label
                print('train:')
                print(example, l) 
                texample, tl = sess.run([valid_image, valid_label])
                print('valid:')
                print(texample,tl)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
            
        coord.request_stop()
        coord.join(threads)