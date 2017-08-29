import getopt
from PIL import Image
import glob
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

import architecture
import Architectures
import evaluate
import Evaluates
import dataset
import dataset_manager
import DatasetManagers
import Datasets
import loss
import Losses
import optimizer
import Optimizers
import utils

ERROR_MSG = 'main.py -a <architecture> -d <dataset> -g <dataset_manager> -l <loss> -o <optimizer>'
def process_args(argv):
    """
    It checks and organizes the arguments in a dictionary.

    Args:
        argv: list containing all parameters and arguments.

    Returns:
        opt_values: dictionary containing parameters as keys and arguments as values.
    """

    try:
        long_opts = ["help", "architecture=", "dataset=",
                     "dataset_manager=", "loss=", "execution_path=",
                     "optimizer=", "evaluate_path=", "evaluate="]
        opts, _ = getopt.getopt(argv, "ha:d:m:g:l:p:o:e:", long_opts)
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
                                utils.arg_validation(arg, architecture.Architecture)
        elif opt in ("-d", "--dataset"):
            opt_values['dataset_name'] = utils.arg_validation(arg, dataset.Dataset)
        elif opt in ("-g", "--dataset_manager"):
            opt_values['dataset_manager_name'] = \
                        utils.arg_validation(arg, dataset_manager.DatasetManager)
        elif opt in ("-e", "--evaluate"):
            opt_values['evaluate_name'] = utils.arg_validation(arg, evaluate.Evaluate)
        elif opt in ("-l", "--loss_name"):
            opt_values['loss_name'] = \
                        utils.arg_validation(arg, loss.Loss)
        elif opt in ("-p", "--execution_path"):
            if os.path.isdir(arg):
                opt_values['execution_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
        elif opt == "--evaluate_path":
            if os.path.isdir(arg):
                opt_values['evaluate_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
        elif opt in ("-o", "--optimizer"):
            opt_values['optimizer_name'] = \
                        utils.arg_validation(arg, optimizer.Optimizer)

    check_needed_parameters(opt_values)
    return opt_values

def check_needed_parameters(parameter_values):
    """This function checks if the needed are given in the main call.

    According with the execution mode this function checks if the needed
    parameteres were defined. If one parameter is missing, this function
    gives an output with the missing parameter and the chosen mode.

    Args:
        parameter_values: Dictionary containing all paramenter names and arguments.
    """
    if parameter_values["execution_mode"] == "train":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name", "optimizer_name"]
    elif parameter_values["execution_mode"] == "restore":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name", "optimizer_name",
                             "execution_path"]
    elif parameter_values["execution_mode"] == "evaluate":
        needed_parameters = ["architecture_name", "execution_path", "evaluate_path",
                             "evaluate_name"]
    else:
        print("Parameters list must contain an execution_mode.")
        sys.exit(2)

    for parameter in needed_parameters:
        if parameter not in parameter_values:
            print("Parameters list must contain an " + parameter + " in the " +\
                  parameter_values["execution_mode"]+" mode.")
            sys.exit(2)

def training(loss_op, optimizer_imp):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss_op: Loss tensor, from loss().
        optimizer_imp: optmizer instance.

    Returns:
        train_op: The Op for training.
    """
    # Get variable list
    network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss_op)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer_imp.minimize(loss_op, global_step=global_step)
    return train_op, global_step


def run_training(opt_values):
    """
    Runs the training

    This function is responsible for instanciating dataset, architecture and loss.abs

    Args:
        opt_values: dictionary containing parameters as keys and arguments as values.

    """
    # Get architecture, dataset and loss name
    arch_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    optimizer_name = opt_values['optimizer_name']
    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)
    dataset_imp = utils.get_implementation(dataset.Dataset, dataset_name)
    loss_imp = utils.get_implementation(loss.Loss, loss_name)
    optimizer_imp = utils.get_implementation(optimizer.Optimizer, optimizer_name)
    time_str = time.strftime("%Y-%m-%d_%H:%M")
    if opt_values["execution_mode"] == "train":
        execution_dir = "Executions/" + dataset_name + "_" + arch_name + "_" + loss_name +\
                    "_" + time_str
        os.makedirs(execution_dir)
    elif opt_values["execution_mode"] == "restore":
        execution_dir = opt_values["execution_path"]
    log(opt_values, execution_dir)
    # Tell TensorFlow that the model will be built into the default Graph.
    graph = tf.Graph()
    with graph.as_default():

        execution_mode = opt_values["execution_mode"]
        # Input and target output pairs.
        architecture_input, target_output = dataset_imp.next_batch_train()

        with tf.variable_scope("model"):
            architecture_output = architecture_imp.prediction(architecture_input, training=True)
        loss_op = loss_imp.evaluate(architecture_output, target_output)
        train_op, global_step = training(loss_op, optimizer_imp)
        # Merge all train summaries and write
        merged = tf.summary.merge_all()
        # Test
        architecture_input_test, target_output_test, init = dataset_imp.next_batch_test()

        with tf.variable_scope("model", reuse=True):
            architecture_output_test = architecture_imp.prediction(architecture_input_test,
                                                                  training=False) # TODO: false?
        loss_op_test = loss_imp.evaluate(architecture_output_test, target_output_test)
        collection_ref = tf.get_collection_ref(tf.GraphKeys.SUMMARIES)
        collection_ref = []
        tf_test_loss = tf.placeholder(tf.float32, shape=(), name="tf_test_loss")
        test_loss = tf.summary.scalar('loss', tf_test_loss)
        # Create summary
        model_dir = os.path.join(execution_dir, "Model")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        summary_dir = os.path.join(execution_dir, "Summary")
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)


        train_writer = tf.summary.FileWriter(summary_dir + '/Train')
        test_writer = tf.summary.FileWriter(summary_dir + '/Test')
        # # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Create a session for running operations in the Graph.
        sess = tf.Session()
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        sess.run(init)
        for i in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            print(i)
        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print(i) 
        sys.exit()


        if execution_mode == "restore":
            # Restore variables from disk.
            model_file_path = os.path.join(model_dir, "model.ckpt")
            saver.restore(sess, model_file_path)
            print("Model restored.")

        step = sess.run(global_step)

        try:

            while True:
                start_time = time.time()
                sess.run(init)
                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                
                loss_value, _, summary = sess.run([loss_op, train_op, merged])
                duration = time.time() - start_time
                train_writer.add_summary(summary, step)

                # Print an overview fairly often.
                if step % 100 == 0:
                    loss_value_sum = 0.0
                    count_test = 0.0
                    try:
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, np.mean(loss_value),
                                                                duration))
                        
                        while True:
                            start_time = time.time()
                            loss_value_test = sess.run(loss_op_test)
                            duration_test = time.time() - start_time
                            count_test = count_test + 1
                            loss_value_sum = loss_value_sum + loss_value_test
                    except tf.errors.OutOfRangeError:
                        print('Done testing')
                        
                    loss_value_test = loss_value_sum / count_test
                    print(loss_value_test)
                    summary_test = sess.run(test_loss,
                                        feed_dict={tf_test_loss:loss_value_test})
                    test_writer.add_summary(summary_test, step)
                    # Save the variables to disk.
                    save_path = saver.save(sess, model_dir + "/model.ckpt")
                    print("Model saved in file: %s" % save_path)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training, %d steps.' % (step))
        finally:
            sess.close()


def log(opt_values, execution_dir): #maybe in another module?
    """
    Stores a .json logfile in execution_dir/logs,
    based on the execution options opt_values.

    Args:
        opt_values: dictionary with the command line arguments of main.py.
        execution_dir: the directory regarding the execution to log.

    Returns:
        Nothing.
    """
    # Get architecture, dataset and loss name
    architecture_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, architecture_name)
    dataset_imp = utils.get_implementation(dataset.Dataset, dataset_name)
    loss_imp = utils.get_implementation(loss.Loss, loss_name)

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

    log_dir = os.path.join(execution_dir, "Logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, 'w') as outfile:
        json.dump(json_data, outfile)

def run_evaluate(opt_values):
    """
    This function performs the evaluation.
    """
    eval_name = opt_values['evaluate_name']
    evaluate_imp = utils.get_implementation(evaluate.Evaluate, eval_name)
    arch_name = opt_values['architecture_name']
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)
    evaluate_imp.eval(opt_values, architecture_imp)



def main(opt_values):
    """
    Main function.
    """
    execution_mode = opt_values['execution_mode']
    if execution_mode in ('train', 'restore'):
        run_training(opt_values)
    elif execution_mode == 'evaluate':
        run_evaluate(opt_values)


if __name__ == "__main__":
    OPT_VALUES = process_args(sys.argv[1:])
    main(OPT_VALUES)
