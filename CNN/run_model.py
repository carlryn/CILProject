import numpy as np
import tensorflow as tf

# our imports
import utils


# Task Parameters
tf.flags.DEFINE_integer("experiment", 0, "0: For training word embeddings; 1: Use trained word embeddings; 2: hidden dim of 1024")

# data params
tf.flags.DEFINE_string("test_file_path", "../data/training/images", "Path to the testing data")
tf.flags.DEFINE_integer("batch_size", 12, "Batch size")
tf.flags.DEFINE_string("save_path","../data/predicted","Where to output the images")
tf.flags.DEFINE_string("path_gt","../data/training/groundtruth", "Path to groundtruth")

# log params
tf.flags.DEFINE_string("log_dir", 'runs/1498673564', "Checkpoint directory (i.e. ../runs/1493459028")
tf.flags.DEFINE_string("graph_file", 'runs/1498673564/model-3000.meta', "Name of meta graph file (i.e. model-400.meta)")

tf.flags.DEFINE_string("device", 'CPU', "CPU or GPU to run on")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))


def main(unused_argv):

    # Load data
    print("Loading data...", flush=True)
    test_data, data_gt, img_names = utils.load_for_testing(FLAGS.test_file_path,
                                                           FLAGS.path_gt
                                                           ,sample=20)
    print("...Data loaded.\n", flush=True)

    # Restore graph & variables
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) if FLAGS.device == 'CPU' else tf.Session()
    tf.train.import_meta_graph(FLAGS.graph_file).restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    # Restore graph & variables
    tf.train.import_meta_graph(FLAGS.graph_file).restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    # Restore ops.
    input_samples_op = tf.get_collection("input_samples_op")[0]
    last_pred = tf.get_collection("predictions")[0]
    # loss = tf.get_collection("loss")[0]
    # mode = tf.get_collection("mode")[0]

    def do_prediction(sess, samples):
        # batches = utils_model.data_iterator_test(samples, FLAGS.batch_size)
        # test_perplexities = []
        batches = [samples[i:i+FLAGS.batch_size]
                   for i in range(0,len(samples),FLAGS.batch_size)]
        pred_images = []
        for batch_samples in batches:
            feed_dict = {input_samples_op: batch_samples}
            # loss_out = sess.run(loss, feed_dict=feed_dict)
            predictions = sess.run(last_pred, feed_dict=feed_dict)
            n,w,h = predictions.shape
            img = np.zeros((w,h))
            for i in range(FLAGS.batch_size):
                for j in range(w):
                    row = predictions[i,j]
                    for k in range(h):
                        dp = predictions[i,j,k]
                        c = np.argmax(dp)
                        img[j,k] = c
                pred_images.append(img)


                        # convert loss to log base 2
            # loss_out = loss_out/np.log(2)

            # Perplexity = 2 ^ avgLoss
            # remove pads
            # pads_mask = np.asarray(np.not_equal(batch_samples[:, 1:], word_ID_map['<pad>']), dtype=np.float32)
            # sequence_word = np.sum(pads_mask, axis=1)
            # perplexity = np.power(2., np.sum(loss_out, axis=1)/sequence_word)
            #
            # test_perplexities.extend(perplexity)

        return pred_images

    pred_images = do_prediction(sess, test_data)[:test_data.shape[0]]
    # Create submission file.
    # with open('group28.perplexityTEST.txt', 'w') as outfile:
    #     for i,img in enumerate(pred_images):
    #         utils.save_image(FLAGS.save_path,test_data,pred_images,img,data_gt[i],img_names[i])
    #         outfile.write(str(i) + "\n")

    utils.save_image(FLAGS.save_path,test_data,pred_images,data_gt,img_names)



if __name__ == '__main__':
    tf.app.run()






