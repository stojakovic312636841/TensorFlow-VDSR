import tensorflow as tf
import numpy as np
import time
import os

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave
)


class VDSR(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 layer,
                 c_dim):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.layer = layer
        self.c_dim = c_dim
        self.build_model()



    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        
        
        self.weights = {
            'w_start': tf.Variable(tf.random_normal([3, 3, self.c_dim, 64], stddev =np.sqrt(2.0/9)), name='w_start'),
            'w_end': tf.Variable(tf.random_normal([3, 3, 64, self.c_dim], stddev=np.sqrt(2.0/9/64)), name='w_end')
        }

        self.biases = {
            'b_start': tf.Variable(tf.zeros([64], name='b_start')),
            'b_end': tf.Variable(tf.zeros([self.c_dim], name='b_end'))
        }

        # Create very deep layer weight and bias
        for  i in range(2, self.layer): #except start and end 
            self.weights.update({'w_%d' % i: tf.Variable(tf.random_normal([3, 3, 64, 64], stddev= np.sqrt(2.0/9/64)), name='w_%d' % i) })
            self.biases.update({'b_%d' % i: tf.Variable(tf.zeros([64], name='b_%d' % i)) })
            
        self.pred = self.model()
        # residul =   labels - images
        self.loss = tf.reduce_mean(tf.square(self.labels - self.images - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint



    def model(self):
        conv = []
        conv.append(tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w_start'], strides=[1,1,1,1], padding='SAME') + self.biases['b_start']))
        for i in range(2, self.layer):
            conv.append(tf.nn.relu(tf.nn.conv2d(conv[i-2], self.weights['w_%d' % i], strides=[1,1,1,1], padding='SAME') + self.biases['b_%d' % i]))
        #conv2 = tf.nn.relu(tf.nn.conv2d(conv[0], self.weights['w_2'], strides=[1,1,1,1], padding='SAME') + self.biases['b_2'])        
        conv_end = tf.nn.conv2d(conv[i-1], self.weights['w_end'], strides=[1,1,1,1], padding='SAME') + self.biases['b_end'] # This layer don't need ReLU
        return conv_end



    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
	# Read image files and make their sub-images and saved them as a h5 file format
        #nx, ny = input_setup(config)

	# get the target(train/test) .h5 file 
        data_dir = checkpoint_dir(config)
        
	# Read h5 format data file
        input_, label_ = read_data(data_dir)

        # Stochastic gradient descent with the standard backpropagation

        # NOTE: learning rate decay
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(config.learning_rate, global_step * config.batch_size, len(input_)*100, 0.1, staircase=True)
        
	# NOTE: Clip gradient
        opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        grad_and_value = opt.compute_gradients(self.loss)
        
        clip = tf.Variable(config.clip_grad, name='clip') 
        capped_gvs = [(tf.clip_by_value(grad, -(clip), clip), var) for grad, var in grad_and_value]

        self.train_op = opt.apply_gradients(capped_gvs, global_step=global_step)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

	#Init all Var
        tf.global_variables_initializer().run()

        counter = 0
        time_ = time.time()

	#To load the checkpoint use to test or pretrain and refine model
        self.load(config.checkpoint_dir)

        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 100 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err ))
                    if counter % 1000 == 0:
                        self.save(config.checkpoint_dir, counter)



    def test(self, config):
	# NOTE : if train, the nx, ny are ingnored
	# Read image files and make their sub-images and saved them as a h5 file format
        nx, ny = input_setup(config)

	# get the test.h5 file 
        data_dir = checkpoint_dir(config)

	# Read h5 format data file
        input_, label_ = read_data(data_dir)

        print("Now Start Testing...")        
	time_ = time.time()    

        result = self.pred.eval({self.images: input_}) + input_
        image = merge(result, [nx, ny], self.c_dim)
        checkimage(merge(result, [nx, ny], self.c_dim))
        
        imsave(image, config.result_dir+'/result.png', config)
	print("time: [%4.4f]" % (time.time()-time_))



    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s_%slayer" % ("vdsr", self.label_size,self.layer)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    


    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "VDSR.model"
        model_dir = "%s_%s_%slayer" % ("vdsr", self.label_size,self.layer)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
	print('save\r')
