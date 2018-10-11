import tensorflow as tf
import numpy as np
import os
from glob import glob
import cv2

train_set_path = './training/'
valid_set_path = './validation/'


# write images and label in tfrecord file and read them out
def encode_to_tfrecords(tfrecords_filename, data_path, sample_num): 
    ''' write into tfrecord file '''
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)

    writer = tf.python_io.TFRecordWriter('./'+tfrecords_filename) # create tfrecords file
    
    for i in range(sample_num):
        images = glob('%sSample%s/*.png' % (data_path, str(i+1).zfill(3)))
        label = [0]* 10
        label[i] = 1
        label = np.array(label, dtype = np.int64)
        for image in images:
            img_raw = cv2.imread(image).astype(np.float32)
            img_raw = img_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=label)),     
                'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString()) 
    
    writer.close()
    return 0



if __name__=='__main__':
    # make train.tfrecord
    train_filename = "train.tfrecords"
    encode_to_tfrecords(tfrecords_filename = train_filename, data_path = train_set_path, sample_num = 10)
    # make validation.tfrecord
    validation_filename = 'validation.tfrecords'
    encode_to_tfrecords(tfrecords_filename = validation_filename , data_path = valid_set_path, sample_num = 10)
    
