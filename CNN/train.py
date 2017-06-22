import os
import time
import datetime
import math
import numpy as np
import tensorflow as tf

# OUR IMPORTS
import model
import utils


import matplotlib.pyplot as plt

# Note that "ops" in the comments refers "tensorflow operations". For the
# details: https://www.tensorflow.org/get_started/get_started

# Data directory
path_train = "../idil/forTraining"
path_test = "../idil/sentences.test"

# Load data

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate (default: 0.5)")

# Training Parameters
tf.flags.DEFINE_integer("learning_rate", 1e-3, "learning rate (default: 1e-3)")
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size")
tf.flags.DEFINE_integer("validation_size", 0.3, "Validation set size as % of initial dataset")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of full passess over whole training data (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_step", 1000,
                        "Evaluate model on validation set after this many steps/iterations (i.e., batches) (default: 500)")

# Log Parameters
tf.flags.DEFINE_integer("print_every_step", 200,
                        "Print training details after this many steps/iterations (i.e., batches) (default: 10)")
tf.flags.DEFINE_integer("checkpoint_every_step", 1000,
                        "Save model after this many steps/iterations (i.e., batches) (default: 1000)")
tf.flags.DEFINE_string("log_dir", "./runs/", "Output directory (default: './runs/')")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
FLAGS.model_dir = os.path.abspath(os.path.join(FLAGS.log_dir, timestamp))
print("Writing to {}\n".format(FLAGS.model_dir))


def main(unused_argv):
    # Get input dimensionality. TODO check best way of defining sizes

    training_data, validation_data = utils.load_train_data(path_train, sample=2)

    # Placeholder variables are used to change the input to the graph.
    # This is where training samples and labels are fed to the graph.
    # These will be fed a batch of training data at each training step
    # using the {feed_dict} argument to the sess.run() call below.
    # input_samples_op = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, SENTENCE_LENGTH],
    #                                   name="input_samples")
    batch_size = FLAGS.batch_size
    w,h = 256,256
    input_samples_op = tf.placeholder(tf.float32, shape=(batch_size,w,h),
                                      name='input_samples')

    # Define embedding vectors matrix.
    # define word embedding matrix ( this is trained as part of the model )
    # with tf.variable_scope('one_hot_encoding'):
    #     one_hot_encoding = tf.get_variable("one_hot_encoding", shape=[FLAGS.vocab_size, FLAGS.vocab_size],
    #                                        trainable=False)

    # Some layers/functions have different behaviours during training and evaluation.
    # If model is in the training mode, then pass True.
    mode = tf.placeholder(tf.bool, name="mode")

    # loss_avg and accuracy_avg will be used to update summaries externally.
    # Since we do evaluation by using batches, we may want average value.
    # (1) Keep counting number of correct predictions over batches.
    # (2) Calculate the average value, evaluate the corresponding summaries
    # by using loss_avg and accuracy_avg placeholders.
    loss_avg = tf.placeholder(tf.float32, name="loss_avg")
    # accuracy_avg = tf.placeholder(tf.float32, name="accuracy_avg")

    # Call the function that builds the network.
    # It returns the logits for the batch [batch_size, sentence_len - 1, embedding_dim].
    logits, labels = model.agricultural_LSTM(input_samples_op,batch_size)

    # Loss calculations: cross-entropy
    with tf.name_scope("cross_entropy_loss"):
        # Takes predictions of the network (logits) and ground-truth labels
        # (input_label_op), and calculates the cross-entropy loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    # Accuracy calculations.
    with tf.name_scope("accuracy"):
        # Return list of predictions (useful for making a submission)
        predictions = tf.argmax(logits, axis=2, name="predictions")
        labels_index = labels
        # Return a float indicating the % of correctly predicted words for the batch
        bool_predictions = tf.equal(predictions, tf.cast(labels_index, dtype=tf.int64))
        # temp = tf.reduce_sum(tf.cast(bool_predictions, tf.float32))
        # temp = tf.Print(temp,[temp, (FLAGS.batch_size*(SENTENCE_LENGTH-1))])
        batch_accuracy = tf.divide(tf.reduce_sum(tf.cast(bool_predictions, tf.float32)),
                                   (FLAGS.batch_size * (SENTENCE_LENGTH - 1)))
        # Number of correct predictions in order to calculate average accuracy afterwards.
        num_correct_predictions = tf.reduce_sum(tf.cast(bool_predictions, tf.int32))

    def do_evaluation(sess, samples):
        '''
        Evaluation function.
        @param sess: tensorflow session object.
        @param samples: input data (numpy tensor)
        @param labels: ground-truth labels (numpy array)
        '''
        # Keep track of this run.
        batches = utils_model.data_iterator(training_data, FLAGS.batch_size)
        counter_accuracy = 0.0
        counter_loss = 0.0
        counter_batches = 0
        for batch_samples in batches:
            counter_batches += 1
            feed_dict = {input_samples_op: batch_samples,
                         mode: False}
            results = sess.run([loss, num_correct_predictions], feed_dict=feed_dict)
            counter_loss += results[0]
            counter_accuracy += results[1]
        return (tf.reduce_mean(counter_loss) / counter_batches, counter_accuracy / (counter_batches * FLAGS.batch_size))

    # Generate a variable to contain a counter for the global training step.
    # Note that it is useful to save/restore network and to set things like annealing schedules for learning rate.
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # Create optimization op.
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))

        # TODO need to add grad clippingi
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, v),
                                             global_step=global_step)

    # For saving/restoring the model.
    # Save important ops by adding them into the collection.
    # Needed to to evaluate our model on the test data after training.
    # See tf.get_collection for details.
    tf.add_to_collection('predictions', logits)
    tf.add_to_collection('input_samples_op', input_samples_op)
    tf.add_to_collection('mode', mode)

    # Create session object
    sess = tf.Session()
    # Add the ops to initialize variables.
    init_op = tf.global_variables_initializer()
    # Actually intialize the variables
    sess.run(init_op)

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and collects
    # data data from it.
    summary_trian_loss = tf.summary.scalar('loss', tf.reduce_mean(loss))
    summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
    summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
    summary_learning_rate = tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    # Group summaries.
    summaries_training = tf.summary.merge([summary_trian_loss, summary_train_acc, summary_learning_rate])
    summaries_evaluation = tf.summary.merge([summary_avg_accuracy])

    # Register summary ops.
    train_summary_dir = os.path.join(FLAGS.model_dir, "summary", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    valid_summary_dir = os.path.join(FLAGS.model_dir, "summary", "validation")
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=3)

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    for epoch in range(1, FLAGS.num_epochs + 1):
        # Generate training batches

        training_batches = utils_model.data_iterator(training_data, FLAGS.batch_size)
        # Training loop.
        for batch_samples in training_batches:
            step = tf.train.global_step(sess, global_step)

            if (step % FLAGS.checkpoint_every_step) == 0:
                ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model'), global_step)
                print("Model saved in file: %s" % ckpt_save_path)

            # This dictionary maps the batch data (as a numpy array) to the
            # placeholder variables in the graph.
            feed_dict = {input_samples_op: batch_samples,
                         mode: True}

            # Run the optimizer to update weights.
            # Note that "train_op" is responsible from updating network weights.
            # Only the operations that are fed are evaluated.
            train_summary, correct_predictions_training, loss_training, _ = sess.run(
                [summaries_training, num_correct_predictions, loss, train_op], feed_dict=feed_dict)

            # Update counters.
            counter_correct_predictions_training += correct_predictions_training
            counter_loss_training += loss_training

            # Write summary data.
            train_summary_writer.add_summary(train_summary, step)

            # Occasionally print status messages.
            if (step % FLAGS.print_every_step) == 0:
                # Calculate average training accuracy.
                accuracy_avg_value_training = counter_correct_predictions_training / (
                    FLAGS.print_every_step * FLAGS.batch_size * (SENTENCE_LENGTH - 1))
                loss_avg_value_training = tf.reduce_mean(counter_loss_training) / (FLAGS.print_every_step)
                # [Epoch/Iteration]
                farming_state = utils.generate_tractor_output()

                print(("[%d/%d] [" + str(farming_state) + "] Accuracy: %.3f, Loss: %.3f") % (
                    epoch, step, accuracy_avg_value_training, loss_avg_value_training.eval(session=sess)))
                counter_correct_predictions_training = 0.0
                counter_loss_training = 0.0
                # Report
                # Note that accuracy_avg and loss_avg placeholders are defined
                # just to feed average results to summaries.
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg: accuracy_avg_value_training,
                                                                           loss_avg: loss_avg_value_training.eval(
                                                                               session=sess)})
                train_summary_writer.add_summary(summary_report, step)

            if (step % FLAGS.evaluate_every_step) == 0:
                # Calculate average validation accuracy.
                (loss_avg_value_validation, accuracy_avg_value_validation) = do_evaluation(sess, validation_data)
                print("[%d/%d] [We have this shit] Accuracy: %.3f, Loss: %.3f" % (
                    epoch, step, accuracy_avg_value_validation, loss_avg_value_validation.eval(session=sess)))
                # Report
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg: accuracy_avg_value_validation,
                                                                           loss_avg: loss_avg_value_validation.eval(
                                                                               session=sess)})
                valid_summary_writer.add_summary(summary_report, step)


if __name__ == '__main__':
    tf.app.run()