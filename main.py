import getopt
import sys
import time
import os

import json
import numpy as np
import tensorflow as tf

import architecture
import Architectures
import dataset
import dataset_manager
import DatasetManagers
import Datasets
import loss
import Losses


def get_implementation(parent_class, child_class_name):
    """Returns a subclass instance.

    It searchs in the subclasses of `parent_class` a class
    named `child_class_name` and return a instance od this class

    Args:
        parent_class: parent class.
        child_class_name: string containing the child class name.

    Returns:
        child_class: instance of child class. `None` if not found.
    """
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return child_class()
    return None

def is_subclass(parent_class, child_class_name):
    """Checks if the parent class has a child with a given name.

    It searchs in the subclasses of `parent_class` a class
    named `child_class_name`. Return True if found and False if not found.

    Args:
        parent_class: parent class.
        child_class_name: string containing the child class name.

    Returns:
        True if found and False if not found.
    """
    for child_class in parent_class.__subclasses__():
        if child_class.__name__ == child_class_name:
            return True
    return False


def arg_validation(arg, cla):
    """Checks if the argument corresponds to a valid class.
    """
    if is_subclass(cla, arg):
        return arg
    else:
        print(str(arg)+" is not a valid " + cla.__module__ + " name.")
        sys.exit(2)


ERROR_MSG = 'main.py -a <architecture> -d <dataset> -g <dataset_manager> -l <loss>'


def process_args(argv):
    """It checks and organizes the arguments in a dictionary.


    """

    try:
        long_opts = ["help", "architecture=", "dataset=",
                     "dataset_manager=", "loss=", "model_path="]
        opts, _ = getopt.getopt(argv, "ha:d:m:g:l:p:", long_opts)
        if opts == []:
            print(ERROR_MSG)
            sys.exit(2)
    except getopt.GetoptError:
        print(ERROR_MSG)
        sys.exit(2)
    opt_values = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(ERROR_MSG)
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg in ('train', 'evaluate', 'restore'):
                opt_values['execution_mode'] = arg
            else:
                print(arg + 'is not a possible execution mode')

                sys.exit(2)
        elif opt in ("-a", "--architecture"):
            opt_values['architecture_name'] = \
                                arg_validation(arg, architecture.Architecture)
        elif opt in ("-d", "--dataset"):
            opt_values['dataset_name'] = arg_validation(arg, dataset.Dataset)
        elif opt in ("-g", "--dataset_manager"):
            opt_values['dataset_manager_name'] = \
                        arg_validation(arg, dataset_manager.DatasetManager)
        elif opt in ("-l", "--loss"):
            opt_values['loss_name'] = \
                        arg_validation(arg, loss.Loss)
        elif opt in ("-p", "--model_path"):
            if os.path.isdir(arg):
                opt_values['model_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
        elif opt in ("-p", "--summary_path"):
            if os.path.isdir(arg):
                opt_values['summary_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
    return opt_values

def training(loss_op, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', tf.reduce_mean(loss_op))
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    return train_op

def check_needed_parameters(parameter_values):
    if parameter_values["execution_mode"] == "train":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name"]
    if parameter_values["execution_mode"] == "restore":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name",
                             "model_path", "summary_path"]
    for parameter in needed_parameters:
        if parameter not in parameter_values:
            print("Parameters list must contain an" + parameter + "in the training mode")
            sys.exit(2)

def run_training(opt_values):
    """Runs the traing given some options

    """
    # Get architecture, dataset and loss name
    arch_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    # Get implementations
    architecture_imp = get_implementation(architecture.Architecture, arch_name)
    dataset_imp = get_implementation(dataset.Dataset, dataset_name)
    loss_imp = get_implementation(loss.Loss, loss_name)
    log(opt_values)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input and target output pairs.
        architecture_input, target_output = dataset_imp.next_batch_train()
        architecture_output = architecture_imp.prediction(architecture_input, training=True)
        print(target_output)
        print(architecture_output)
        print(architecture_input)
        loss_op = loss_imp.evaluate(architecture_output, target_output)
        print(loss_op)
        train_op = training(loss_op, 10**(-4))
        # Create summary
        time_str = time.strftime("%Y-%m-%d_%H:%M")
        if opt_values["execution_mode"] == "train":
            summary_dir = "Summaries/" + dataset_name + "_" + arch_name + "_" + loss_name +\
                            "_" + time_str
            os.makedirs(summary_dir)
        elif opt_values["execution_mode"] == "restore":
            summary_dir = opt_values["summary_path"]
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir + '/train')
        test_writer = tf.summary.FileWriter(summary_dir + '/test')
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        if opt_values["execution_mode"] == "train":
            model_dir = "Models/" + dataset_name + "_" + arch_name + "_" + loss_name +\
                        "_" + time_str
            os.makedirs(model_dir)
        elif opt_values["execution_mode"] == "restore":
            model_dir = opt_values["model_path"]
            
        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            if opt_values["execution_mode"] == "restore":
                # Restore variables from disk.
                saver.restore(sess, "/tmp/model.ckpt")
                print("Model restored.")
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                summary, loss_value, _ = sess.run([merged, loss_op, train_op])
                duration = time.time() - start_time
                train_writer.add_summary(summary, step)
                # Print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, np.mean(loss_value),
                                                               duration))
                    # Save the variables to disk.
                    save_path = saver.save(sess, model_dir + "/model.ckpt")
                    print("Model saved in file: %s" % save_path)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' %
                  (dataset_imp.config_dict["num_epochs"], step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

def log(opt_values): #maybe in another module?
    # Get architecture, dataset and loss name
    architecture_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    # Get implementations
    architecture_imp = get_implementation(architecture.Architecture, architecture_name)
    dataset_imp = get_implementation(dataset.Dataset, dataset_name)
    loss_imp = get_implementation(loss.Loss, loss_name)

    today = time.strftime("%Y-%m-%d_%H:%M")
    log_name = dataset_name + "_" + architecture_name + "_" + loss_name + "_" +\
               opt_values['execution_mode'] + "_" + today +".json"
    json_data = {}
    if hasattr(architecture_imp, 'config_dict'):
        json_data["architecture"] = architecture_imp.config_dict
    if hasattr(loss_imp, 'config_dict'):
        json_data["loss"] = loss_imp.config_dict
    if hasattr(dataset_imp, 'config_dict'):
        json_data["architecture"] = dataset_imp.config_dict
    json_data["execution_mode"] = opt_values['execution_mode']
    json_data["architecture_name"] = architecture_name
    json_data["dataset_name"] = dataset_name
    json_data["loss_name"] = loss_name

    logdir = os.path.abspath("Logs/") #TODO(Rael): Use options instead
    log_path = os.path.join(logdir, log_name)
    with open(log_path, 'w') as outfile:
        json.dump(json_data, outfile)

if __name__ == "__main__":
    OPT_VALUES = process_args(sys.argv[1:])
    EXECUTION_MODE = OPT_VALUES['execution_mode']
    check_needed_parameters(OPT_VALUES)
    if EXECUTION_MODE == 'train':
        run_training(OPT_VALUES)
