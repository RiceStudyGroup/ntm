from utils import OmniglotDataLoader, one_hot_decode, five_hot_decode
import tensorflow as tf
import argparse
import numpy as np
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug


class Config():

    def __init__(self):
        self.mode = "train"
        self.restore_training = False
        self.debug = False
        self.label_type = "one_hot"
        self.n_classes = 5
        self.seq_length = 50
        self.augment = True
        self.model = "MANN"
        self.read_head_num = 4
        self.batch_size = 16
        self.num_epoches = 100000
        self.learning_rate = 1e-3
        self.rnn_size = 200
        self.image_width = 20
        self.image_height = 20
        self.rnn_num_layers = 1
        self.memory_size = 128
        self.memory_vector_dim = 40
        self.shift_range = 1
        self.write_head_num = 1
        self.test_batch_num = 100
        self.n_train_classes = 1200
        self.n_test_classes = 423
        self.save_dir = './save/one_shot_learning'
        self.tensorboard_dir = './summary/one_shot_learning'

        pass


def main():
    args = Config()
    if args.mode == 'train':
        train(args)


def train(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(
        data_dir="/Users/xavier.qiu/Documents/ricecourse/comp590Research/data/omniglot/images_background/",
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    with tf.Session() as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        print(args)
        print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")
        for b in range(args.num_epoches):

            # Test

            if b % 100 == 0:
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                              type='test',
                                                              augment=args.augment,
                                                              label_type=args.label_type)
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                # with open('state_long.txt', 'w') as f:
                #     print(state_list, file=f)
                accuracy = test_f(args, y, output)
                for accu in accuracy:
                    print('%.4f' % accu, end='\t')
                print('%d\t%.4f' % (b, learning_loss))

            # Save model

            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # Train

            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='train',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            sess.run(model.train_op, feed_dict=feed_dict)


if __name__ == '__main__':
    main()
