#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     export.py
#
# Description:  delete useless nodes
# Version:      1.0
# Created:      2020/3/20 15:46
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#


import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.framework import list_variables, load_variable
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "checkpoint path")
flags.DEFINE_string("export_path", None, "output node names")
flags.DEFINE_string("tpu_address", None, "tpu address")


def export(checkpoint, export_path, tpu_address):
    output_ckpt = os.path.join(export_path, "model.ckpt-best")
    tf.reset_default_graph()
    clear_devices = True
    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=clear_devices)
    with tf.Session(tpu_address) as sess:
        saver.restore(sess, checkpoint)
        my_vars = []
        for var in tf.global_variables():
            print(var.name)
            # if "adam_v" not in var.name and "adam_m" not in var.name:
            #     my_vars.append(var)
        # new_saver = tf.train.Saver(my_vars)
        # new_saver.save(sess, output_ckpt)


if __name__ == '__main__':
    export(FLAGS.checkpoint, FLAGS.export_path, FLAGS.tpu_address)












# import os
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# from tensorflow.contrib.framework import list_variables, load_variable
# flags = tf.flags
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_string("checkpoint", None, "checkpoint path")
# flags.DEFINE_string("output_node_names", None, "output node names")
# flags.DEFINE_string("pb_name", "frozen_model.pb", "pb name, default: frozen_model.pb")
# flags.DEFINE_bool("do_ensemble", False, "ensemble model or not")
#
# def freeze_graph(checkpoint, output_node_names=None, pb_name="frozen_model.pb", do_ensemble=False):
#     output_dir = os.path.dirname(checkpoint)
#     output_graph = os.path.join(output_dir, pb_name)
#     tf.reset_default_graph()
#     clear_devices = False
#     saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=clear_devices)
#     graph = tf.get_default_graph()
#     input_graph_def = graph.as_graph_def()
#     if not output_node_names:
#         output_node_names = "input_ids,input_mask,segment_ids,answer_class/cls_logits,start_top_log_probs,start_top_index,end_top_log_probs,end_top_index"
#         if do_ensemble:
#             output_node_names += ",start_logits/start_log_probs_st,answer_class/cls_logits_st"
#     # vars = []
#     # for var_name, _ in  list_variables(checkpoint):
#     #     var = load_variable(checkpoint, var_name)
#     #     vars.append(var)
#     # saver = tf.train.Saver(var_list=vars)
#
#     with tf.Session("grpc://10.58.189.194:8470") as sess:
#         saver.restore(sess, checkpoint)
#         output_graph_def = graph_util.convert_variables_to_constants(
#             sess,
#             input_graph_def,
#             output_node_names.split(',')
#         )
#         with tf.gfile.GFile(output_graph, 'wb') as f:
#             f.write(output_graph_def.SerializeToString())
#
#
# if __name__ == '__main__':
#     freeze_graph(FLAGS.checkpoint, FLAGS.output_node_names, FLAGS.pb_name, FLAGS.do_ensemble)
