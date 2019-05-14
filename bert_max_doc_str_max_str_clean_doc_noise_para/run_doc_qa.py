#!/usr/bin/env python2
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import math
import os
import random
import time
import six
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
from scipy.special import logsumexp


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("filter_null_doc", True,
                  "Whether to filter out no-answer document.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("max_num_doc_feature", 12,
                     "Max number of document features allowed.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_float(
    "teacher_temperature", 1.0,
    "A knob to sharpen or flatten the teacher distribution.")

flags.DEFINE_float(
    "local_obj_alpha", 0.0,
    "Trades off between clean global and noisy local objectives.")

flags.DEFINE_bool(
    "label_cleaning", True,
    "Performs global label cleaning.")

flags.DEFINE_bool(
    "marginalize", False,
    "If true we marginalize over multiple possible answers in training.")

flags.DEFINE_bool(
    "debug", False,
    "If true we process a tiny dataset.")

flags.DEFINE_bool(
    "train_first_answer", True,
    "If true we train using just the first possible answer.")

flags.DEFINE_bool(
    "posterior_distillation", False,
    "If true we distill teacher supervision in training.")

flags.DEFINE_string(
    "pd_loss", "sqerr",
    "The distance function between teacher and student predictions.")

flags.DEFINE_bool(
    "doc_normalize", False,
    "If true, performs document-level normalization."
)

flags.DEFINE_bool(
    "add_null_simple", False,
    "If true, we add the null span as possible in all cases.")


flags.DEFINE_integer("max_short_answers", 10,
                     "The maximum number of distinct short answer positions.")

flags.DEFINE_integer("max_num_answer_strings", 80,
                     "The maximum number of distinct short answer strings.")

flags.DEFINE_integer("max_paragraph", 4,
                     "The maximum numbr of paragraph allowed in a document.")

flags.DEFINE_string("no_answer_string", "",
                    "The string is used for as no-answer string.")

## Modified configuration parameters.
flags.DEFINE_string("device", "gpu",
                    "The main device is used for training.")

flags.DEFINE_integer("num_cpus", 4,
                     "The number of cpus is used.")

flags.DEFINE_string("initializer", "Xavier",
                    "The initializer is used for parameters.")

flags.DEFINE_bool(
    "shuffle_data", True,
    "If True, shuffles the training data locally.")


class SquadExample(object):
    """A single training/test example for simple sequence classification.
    Each example has one question, the corresponding answer(s), and a single
     paragraph as the evidence.

     For examples without an answer, the start and end position are -1.
    """

    __slots__ = ('qas_id', 'question_text', 'doc_tokens',
                 'orig_answer_text', 'start_position', 'end_position',
                 'orig_answer_text_list', 'start_position_list',
                 'end_position_list', 'is_impossible', 'qid')

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 qid=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False,
                 orig_answer_text_list=None,
                 start_position_list=None,
                 end_position_list=None):
        self.qas_id = qas_id

        if qid:
            # For TriviaQA, there is question id which can be used for document
            # level normalization.
            self.qid = qid
        else:
            self.qid = qas_id

        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

        self.is_impossible = is_impossible

        # This is an extension to support weak supervisions.
        self.orig_answer_text_list = orig_answer_text_list
        self.start_position_list = start_position_list
        self.end_position_list = end_position_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        if self.start_position and self.start_position_list:
            s += ", start_positions: [{0}]".format(
                ",".join(self.start_position_list))
        else:
            s += ", start_positions: []"
        return s


def convert_string_key_to_int_key(orig_dict):
    """Converts the key from string to integer."""
    return dict([(int(key), val) for key, val in orig_dict.items()])


class InputFeatures(object):
    """A single set of features of data."""

    __slots__ = ('unique_id', 'example_index', 'doc_span_index', 'tokens',
                 'token_to_orig_map', 'token_is_max_context', 'input_ids',
                 'input_mask', 'segment_ids', 'qid', 'start_position',
                 'end_position', 'is_impossible', 'start_position_list',
                 'end_position_list', 'position_mask', 'answer_index_list',
                 'num_answer',)

    __dict_attr__ = ('token_to_orig_map', 'token_is_max_context',)

    def __init__(self,
                 unique_id=None,
                 example_index=None,
                 doc_span_index=None,
                 tokens=None,
                 token_to_orig_map=None,
                 token_is_max_context=None,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,
                 qid=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 start_position_list=None,
                 end_position_list=None,
                 answer_index_list=None,
                 load_from_json=False):

        if load_from_json:
            return

        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

        # Adds the qid for normalization.
        if not qid:
            raise ValueError("qid can not be None!")

        self.qid = qid

        # Pads the raw start and end positions.
        (self.start_position_list, self.end_position_list,
         self.position_mask, self.answer_index_list
        ) = InputFeatures.make_fixed_length(
            start_position_list, end_position_list, answer_index_list
        )

        # Truncates possible answers if there are too many possibles.
        self.num_answer = 0
        if self.start_position_list:
            self.num_answer = sum(self.position_mask)

    def to_json(self):
        """Serializes the object into the json form."""
        return json.dumps(dict([
            (attr, getattr(self, attr))
            for attr in self.__slots__
        ]))

    def parse_from_json(self, json_string):
        """Parses the object from a given json string."""
        attr_dict = json.loads(json_string)
        for attr in self.__slots__:
            if attr in self.__dict_attr__:
                val = convert_string_key_to_int_key(attr_dict[attr])
            else:
                val = attr_dict[attr]
            setattr(self, attr, val)
        return True

    @staticmethod
    def load_from_json(json_string):
        """Loads the object from json string."""
        obj = InputFeatures(load_from_json=True)
        obj.parse_from_json(json_string)
        return obj

    @staticmethod
    def make_fixed_length(start_position_list, end_position_list,
                          answer_index_list):
        """Returns three fixed length lists: start, end, and mask."""
        if start_position_list is None:
            return None, None, None, None

        # Truncates possible answers if there are too many possibles.
        len_ans = min(len(start_position_list), FLAGS.max_short_answers)

        # Initializes all lists.
        position_mask = [1 if kk < len_ans else 0
                         for kk in xrange(FLAGS.max_short_answers)]
        start_positions = list(position_mask)
        end_positions = list(position_mask)
        answer_indices = list(position_mask)

        for ii, (start, end, ans_ind) in enumerate(itertools.islice(
                itertools.izip(start_position_list, end_position_list,
                               answer_index_list), len_ans)):
            start_positions[ii] = start
            end_positions[ii] = end
            answer_indices[ii] = ans_ind

        if not start_position_list:
            raise ValueError("No answer positions!")

        return start_positions, end_positions, position_mask, answer_indices


class DocInputFeatures(object):
    """A single set of features of data."""

    __slots__ = ('unique_id', 'feature_list', 'num_feature', 'qid',
                 'num_unique_answer_str', 'answer_string_to_id',)

    def __init__(self, unique_id=None, feature_list=None, num_feature=None,
                 qid=None, answer_string_to_id=None, load_from_json=False):

        if load_from_json:
            return

        self.unique_id = unique_id
        self.feature_list = feature_list
        self.num_feature = num_feature
        self.num_unique_answer_str = len(answer_string_to_id)
        self.answer_string_to_id = answer_string_to_id

        if self.num_unique_answer_str < 2:
            tf.logging.info("unique_id={0} has {1} unique answer string".format(
                unique_id, self.num_unique_answer_str
            ))

            for feature in feature_list:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (feature.unique_id))
                tf.logging.info("example_index: %s" % (feature.example_index))
                tf.logging.info("doc_span_index: %s" % (feature.doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [x for x in feature.tokens]))
                if feature.is_impossible:
                    tf.logging.info("impossible example")
                tf.logging.info("start_position_list: %s" % " ".join(
                    [str(x) for x in feature.start_position_list]))
                tf.logging.info("end_position_list: %s" % " ".join(
                    [str(x) for x in feature.end_position_list]))
                for start, end, ans_ind in zip(feature.start_position_list,
                                               feature.end_position_list,
                                               feature.answer_index_list):
                    answer_text = " ".join(
                        feature.tokens[start:(end + 1)])
                    tf.logging.info("start_position: %d" % (start))
                    tf.logging.info("end_position: %d" % (end))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))
                    tf.logging.info("answer index: %d" % ans_ind)
            tf.logging.info("answer_string_to_id: %s" %
                            json.dumps(answer_string_to_id))

            raise ValueError("There should at least two unique answer strings!")

        # Adds the qid for normalization.
        if not qid:
            raise ValueError("qid can not be None!")

        self.qid = qid

    def to_json(self):
        """Serializes the object into the json form."""
        attr_val_list = []
        for attr in self.__slots__:
            if attr == 'feature_list':
                continue
            attr_val_list.append((attr, getattr(self, attr)))

        attr_val_list.append((
            'feature_list', [feature.to_json() for feature in self.feature_list]
        ))
        return json.dumps(dict(attr_val_list))

    def parse_from_json(self, json_string):
        """Parses the object from a given json string."""
        attr_dict = json.loads(json_string)
        for attr in self.__slots__:
            if attr == 'feature_list':
                val = [
                    InputFeatures.load_from_json(feature_json)
                    for feature_json in attr_dict[attr]
                ]
            else:
                val = attr_dict[attr]
            setattr(self, attr, val)
        return True

    @staticmethod
    def load_from_json(json_string):
        """Loads the object from json string."""
        obj = DocInputFeatures(load_from_json=True)
        obj.parse_from_json(json_string)
        return obj


def get_position_lists(answers_list, char_to_word_offset, doc_tokens):
    """Reads a list of answers and creates lists for the unique ones."""
    start_position_list = []
    end_position_list = []
    answer_text_list = []

    for answer in answers_list:
        orig_answer_text = answer["text"]
        answer_offset = int(answer["answer_start"])
        answer_length = len(orig_answer_text)
        start_position = char_to_word_offset[answer_offset]
        end_position = char_to_word_offset[answer_offset + answer_length - 1]

        # Skips duplicate answers.
        if (start_position in start_position_list and
                end_position in end_position_list):
            continue

        # Checks whether the answer text can be recovered.
        # If not, skips the current answer.
        actual_text = " ".join(
            doc_tokens[start_position:(end_position + 1)])
        cleaned_answer_text = " ".join(
            tokenization.whitespace_tokenize(orig_answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            tf.logging.warning(
                "Could not find answer: '%s' vs. '%s'",
                actual_text, cleaned_answer_text)
            continue

        start_position_list.append(start_position)
        end_position_list.append(end_position)
        answer_text_list.append(orig_answer_text)

    return start_position_list, end_position_list, answer_text_list


def squad_example_generator(input_data, is_training):
    """A generator for squad example."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    counter = 0
    stop_processing = False
    for entry in input_data:

        if stop_processing:
            break

        for paragraph in entry["paragraphs"]:
            if FLAGS.debug and counter >= 1000:
                tf.logging.info("[Debugging]: only keeps 1000 examples.")
                stop_processing = True
                break
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                qid = None
                if "qid" in qa:
                    qid = qa["qid"]
                else:
                    # TODO(chenghao): This is debug report for TriviaQA.
                    raise ValueError("No qid in qa")

                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                start_position_list = None
                end_position_list = None
                orig_answer_text_list = None
                if is_training:
                    if FLAGS.version_2_with_negative:
                        if not (type(qa["is_impossible"]) is bool):
                            raise ValueError("is_impossible is not bool")

                        is_impossible = qa["is_impossible"]

                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = int(answer["answer_start"])
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[
                            answer_offset + answer_length - 1]

                        # For weak-supervision datasets, there might be multiple
                        # answers.
                        (start_position_list, end_position_list,
                         orig_answer_text_list) = get_position_lists(
                             qa["answers"], char_to_word_offset, doc_tokens
                         )

                        if not start_position_list:
                            continue

                        # Only add answers where the text can be exactly
                        # recovered from the document. If this CAN'T happen it's
                        # likely due to weird Unicode stuff so we will just skip
                        # the example.
                        #
                        # Note that this means for training mode, every example
                        # is NOT guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning(
                                "Could not find answer: '%s' vs. '%s'",
                                actual_text, cleaned_answer_text)
                            continue

                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = FLAGS.no_answer_string

                        start_position_list = [-1]
                        end_position_list = [-1]
                        orig_answer_text_list = [FLAGS.no_answer_string]

                # TODO(chenghao): Fix this.
                counter += 1
                yield SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    qid=qid,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    orig_answer_text_list=orig_answer_text_list,
                    start_position_list=start_position_list,
                    end_position_list=end_position_list
                )


def read_squad_examples_from_generator(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    tf.logging.info("Reading examples in using generator!")

    examples = [
        example for example in squad_example_generator(input_data, is_training)
    ]
    tf.logging.info("Done Reading.")

    return examples


_DocSquadExample = collections.namedtuple(
    "DocSquadExample", ["qid", "example_list"]
)


def read_doc_squad_examples_from_generator(input_file, is_training):
    """Read a SQuAD json file into a list of DocSquadExample."""
    examples = read_squad_examples_from_generator(input_file, is_training)

    keyfunc = lambda x: x.qid

    # Groups the example by qid.
    doc_examples = [
        _DocSquadExample(qid=qid, example_list=list(group))
        for qid, group in itertools.groupby(
            sorted(examples, key=keyfunc), key=keyfunc)
    ]

    return doc_examples


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    stop_processing = False
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            if FLAGS.debug and len(examples) >= 1000:
                tf.logging.info("[Debugging]: only keeps 1000 examples.")
                stop_processing = True
                break
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                qid = None
                if "qid" in qa:
                    qid = qa["qid"]
                else:
                    # TODO(chenghao): This is debug report for TriviaQA.
                    raise ValueError("No qid in qa")

                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                start_position_list = None
                end_position_list = None
                orig_answer_text_list = None
                if is_training:
                    if FLAGS.version_2_with_negative:
                        if not (type(qa["is_impossible"]) is bool):
                            raise ValueError("is_impossible is not bool")

                        is_impossible = qa["is_impossible"]

                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = int(answer["answer_start"])
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[
                            answer_offset + answer_length - 1]

                        # For weak-supervision datasets, there might be multiple
                        # answers.
                        (start_position_list, end_position_list,
                         orig_answer_text_list) = get_position_lists(
                             qa["answers"], char_to_word_offset, doc_tokens
                         )

                        if not start_position_list:
                            continue

                        # Only add answers where the text can be exactly
                        # recovered from the document. If this CAN'T happen it's
                        # likely due to weird Unicode stuff so we will just skip
                        # the example.
                        #
                        # Note that this means for training mode, every example
                        # is NOT guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning(
                                "Could not find answer: '%s' vs. '%s'",
                                actual_text, cleaned_answer_text)
                            continue


                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                        start_position_list = [-1]
                        end_position_list = [-1]
                        orig_answer_text_list = [""]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    qid=qid,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    orig_answer_text_list=orig_answer_text_list,
                    start_position_list=start_position_list,
                    end_position_list=end_position_list
                )
                examples.append(example)

        if stop_processing:
            break

    return examples


def convert_doc_examples_to_feature_list(
        doc_examples, tokenizer, max_seq_length, doc_stride, max_query_length,
        is_training, output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    start_id = 1000000000
    doc_id = start_id
    unique_offset = 0

    num_unique_str_stats = []

    null_doc_cnt = 0
    null_qid_list = []

    if is_training and FLAGS.filter_null_doc:
        tf.logging.info("Filtering null documents for training.")

    for doc_example in doc_examples:
        answer_string_to_id = collections.defaultdict(int)

        # Reserves the 0 index for no answer string.
        # Non-empty answers will have indices starting from 1.
        answer_string_to_id[FLAGS.no_answer_string] = 0

        feature_list = [
            feature for feature in convert_examples_to_feature_list_within_doc(
                doc_example.example_list, tokenizer, max_seq_length, doc_stride,
                max_query_length, is_training, unique_offset,
                answer_string_to_id
            )
        ]

        if len(answer_string_to_id) < 2:
            null_doc_cnt += 1
            null_qid_list.append(feature_list[0].qid)

            # We throws out examples without any answer string in the document.
            if is_training and FLAGS.filter_null_doc:
                continue

        unique_offset += len(feature_list)

        doc_feature = DocInputFeatures(
            unique_id=doc_id,
            feature_list=feature_list,
            num_feature=len(feature_list),
            answer_string_to_id=answer_string_to_id,
            qid=doc_example.qid
        )

        num_unique_str_stats.append(len(answer_string_to_id))

        output_fn(doc_feature.to_json())
        output_fn('\n')

        doc_id += 1

    tf.logging.info("Max num of unique answer string: {0}".format(
        np.max(num_unique_str_stats)))
    tf.logging.info("Min num of unique answer string: {0}".format(
        np.min(num_unique_str_stats)))
    tf.logging.info("Min num of unique answer string: {0}".format(
        np.mean(num_unique_str_stats)))
    tf.logging.info("Number of null answer doc: {0}".format(null_doc_cnt))
    tf.logging.info("Null answer qids:{0}".format(null_qid_list))

    return unique_offset


def convert_examples_to_feature_list_within_doc(
        examples, tokenizer, max_seq_length, doc_stride, max_query_length,
        is_training, unique_offset, answer_string_to_id):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000 + unique_offset

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        tok_start_position_list = []
        tok_end_position_list = []

        # Registers answer strings of the current examples.
        tok_answer_index_list = []

        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
            tok_start_position_list = [0]
            tok_end_position_list = [0]
            tok_answer_index_list = [0]

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]

            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

            # Iterates over all possible answer positions.
            for (start_index, end_index, orig_ans_txt) in zip(
                    example.start_position_list,
                    example.end_position_list,
                    example.orig_answer_text_list):
                tok_start_pos = orig_to_tok_index[start_index]
                tok_end_pos = len(all_doc_tokens) - 1
                if end_index < len(example.doc_tokens) - 1:
                    tok_end_pos = orig_to_tok_index[end_index + 1] - 1

                # Improves both start and end positions.
                (tok_start_pos_improved,
                 tok_end_pos_improved) = _improve_answer_span(
                     all_doc_tokens, tok_start_pos, tok_end_pos, tokenizer,
                     orig_ans_txt)

                ans_text_improved = " ".join(all_doc_tokens[
                    tok_start_pos_improved:(tok_end_pos_improved + 1)])

                if ans_text_improved not in answer_string_to_id:
                    # Only keeps answer strings up to the upperbound.
                    # TODO(chenghao): Changes this to frequency-based.
                    if len(answer_string_to_id) <= FLAGS.max_num_answer_strings:
                        answer_string_to_id[ans_text_improved] = len(
                            answer_string_to_id)
                    else:
                        tf.logging.info(
                            "qid %s has more than %d short answers" % (
                                example.qid, FLAGS.max_num_answer_strings
                            ))

                ans_str_index = answer_string_to_id.get(ans_text_improved, 0)

                if ans_text_improved and ans_str_index == 0:
                    tf.logging.info("Drops answer %s for qid %s" % (
                        ans_text_improved, example.qid
                    ))

                tok_start_position_list.append(tok_start_pos_improved)
                tok_end_position_list.append(tok_end_pos_improved)
                tok_answer_index_list.append(ans_str_index)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[
                    len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            start_position_list = None
            end_position_list = None
            answer_index_list = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an
                # answer, we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                start_position_list = []
                end_position_list = []
                answer_index_list = []

                for (tok_start_pos, tok_end_pos, tok_ans_index) in zip(
                        tok_start_position_list, tok_end_position_list,
                        tok_answer_index_list):
                    # If the answer is out of range, skips this pair.
                    if not (tok_start_pos >= doc_start and
                            tok_end_pos <= doc_end and
                            tok_start_pos <= tok_end_pos):
                        continue

                    # Computes the start and end positions with the offset.
                    doc_offset = len(query_tokens) + 2
                    start_position_list.append(
                        tok_start_pos - doc_start + doc_offset
                    )
                    end_position_list.append(
                        tok_end_pos - doc_start + doc_offset
                    )

                    answer_index_list.append(tok_ans_index)

                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end and
                        tok_start_position <= tok_end_position):
                    out_of_span = True

                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # TODO(chenghao): This part needs to be cleaned up.
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]
                answer_index_list = [0]

            if is_training and not start_position_list:
                print("No answer positions!")
                start_position_list = [0]
                end_position_list = [0]
                answer_index_list = [0]

            # TODO(chenghao): This is kept for the purpose of making training
            # and evaluation consistent.
            if not is_training:
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]
                answer_index_list = [0]

            if unique_offset < 1 and example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join(
                    [str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training:
                    tf.logging.info("start_position_list: %s" % " ".join(
                        [str(x) for x in start_position_list]))
                    tf.logging.info("end_position_list: %s" % " ".join(
                        [str(x) for x in end_position_list]))
                    for start, end in zip(start_position_list,
                                          end_position_list):
                        answer_text = " ".join(
                            tokens[start:(end + 1)])
                        tf.logging.info("start_position: %d" % (start))
                        tf.logging.info("end_position: %d" % (end))
                        tf.logging.info("answer: %s" % (
                            tokenization.printable_text(answer_text)))
                    tf.logging.info("answer_string_to_id: %s" %
                                    json.dumps(answer_string_to_id))

                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                qid=example.qid,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                start_position_list=start_position_list,
                end_position_list=end_position_list,
                answer_index_list=answer_index_list
            )

            unique_id += 1

            # This is originally inside the constructor of InputFeatures.
            # For each paragraph, there should at least one answer.
            if is_training and feature.num_answer < 1:
                raise ValueError(
                    "Each paragraph should have at least one answer!")

            yield feature


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, unique_id_to_qid=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        tok_start_position_list = []
        tok_end_position_list = []
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
            tok_start_position_list = [-1]
            tok_end_position_list = [-1]

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]

            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

            # Iterates over all possible answer positions.
            for (start_index, end_index, orig_ans_txt) in zip(
                    example.start_position_list,
                    example.end_position_list,
                    example.orig_answer_text_list):
                tok_start_pos = orig_to_tok_index[start_index]
                tok_end_pos = len(all_doc_tokens) - 1
                if end_index < len(example.doc_tokens) - 1:
                    tok_end_pos = orig_to_tok_index[end_index + 1] - 1

                # Improves both start and end positions.
                (tok_start_pos_improved,
                 tok_end_pos_improved) = _improve_answer_span(
                     all_doc_tokens, tok_start_pos, tok_end_pos, tokenizer,
                     orig_ans_txt)

                tok_start_position_list.append(tok_start_pos_improved)
                tok_end_position_list.append(tok_end_pos_improved)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[
                    len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            start_position_list = None
            end_position_list = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an
                # answer, we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                start_position_list = []
                end_position_list = []

                for tok_start_pos, tok_end_pos in zip(tok_start_position_list,
                                                      tok_end_position_list):
                    # If the answer is out of range, skips this pair.
                    if not (tok_start_pos >= doc_start and
                            tok_end_pos <= doc_end and
                            tok_start_pos <= tok_end_pos):
                        continue

                    # Computes the start and end positions with the offset.
                    doc_offset = len(query_tokens) + 2
                    start_position_list.append(
                        tok_start_pos - doc_start + doc_offset
                    )
                    end_position_list.append(
                        tok_end_pos - doc_start + doc_offset
                    )

                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end and
                        tok_start_position <= tok_end_position):
                    out_of_span = True

                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
                start_position_list = [0]
                end_position_list = [0]

            if is_training and not start_position_list:
                print("No answer positions!")
                start_position_list = [0]
                end_position_list = [0]

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join(
                    [str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training:
                    tf.logging.info("start_position_list: %s" % " ".join(
                        [str(x) for x in start_position_list]))
                    tf.logging.info("end_position_list: %s" % " ".join(
                        [str(x) for x in end_position_list]))
                    for start, end in zip(start_position_list,
                                          end_position_list):
                        answer_text = " ".join(
                            tokens[start:(end + 1)])
                        tf.logging.info("start_position: %d" % (start))
                        tf.logging.info("end_position: %d" % (end))
                        tf.logging.info("answer: %s" % (
                            tokenization.printable_text(answer_text)))

                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("answer: %s" % (
                        tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                qid=example.qid,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                start_position_list=start_position_list,
                end_position_list=end_position_list
            )

            # Run callback
            output_fn(feature)

            if unique_id_to_qid is not None:
                if example.qid:
                    unique_id_to_qid[unique_id] = example.qid
                else:
                    raise ValueError("When unique_id_to_qid is required,"
                                     "example.qid can not be None!")

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can
    # match the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a
    # single token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _build_initializer(initializer):
    """Builds initialization method for the TF model."""
    if initializer == 'Uniform':
        tf.logging.info('Using random_uniform_initializer')
        tf_initializer = tf.random_uniform_initializer(
            -0.1, 0.1, dtype=tf.float32
        )
    elif initializer == 'Gaussian':
        tf.logging.info('Using truncated_normal_initializer')
        tf_initializer = tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32
        )
    elif initializer == 'Xavier':
        tf.logging.info('Using xavier_initializer')
        tf_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32
        )
    else:
        raise ValueError('Unknown initializer {0}!'.format(initializer))

    return tf_initializer


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))

        # TODO(chenghao): We can have the single input feature as a single
        # document with FLAGS.max_paragraph.
        unique_ids = features["unique_ids"]

        # Looks up document normalization scores.
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def one_hot_answer_positions(position_list, position_mask, depth):
                position_tensor = tf.one_hot(
                    position_list, depth, dtype=tf.float32)
                position_masked = position_tensor * tf.cast(
                    tf.expand_dims(position_mask, -1), dtype=tf.float32
                )
                onehot_positions = tf.reduce_max(position_masked, axis=1)
                return onehot_positions

            def compute_marginalized_loss(logits, position_list, answer_masks):
                position_tensor = one_hot_answer_positions(
                    position_list, answer_masks, seq_length)
                z_all = tf.reduce_logsumexp(logits, -1)

                z_positive = tf.reduce_logsumexp(
                    logits + tf.log(position_tensor), -1)
                loss = tf.reduce_mean(z_all - z_positive)
                return loss

            def compute_pd_loss(logits, position_list, answer_masks):
                position_tensor = one_hot_answer_positions(
                    position_list, answer_masks, seq_length
                    )
                z_all = tf.reduce_logsumexp(logits, -1)

                masked_logits = tf.stop_gradient(
                    logits + tf.log(position_tensor)
                ) * FLAGS.teacher_temperature

                fixed_teacher_dist = tf.nn.softmax(masked_logits, axis=-1)
                if FLAGS.pd_loss == "sqerr":
                    model_dist = tf.nn.softmax(logits, axis=-1)
                    return tf.reduce_sum(tf.squared_difference(
                        tf.sqrt(fixed_teacher_dist),
                        tf.sqrt(model_dist)
                    ))
                else:
                    return tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=fixed_teacher_dist,
                            logits=logits,
                            axis=-1)
                    )

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            if FLAGS.train_first_answer:
                # TODO(chenghao): This is a debug flag.
                raise ValueError("")
                start_positions = features["start_positions"]
                end_positions = features["end_positions"]

                start_loss = compute_loss(start_logits, start_positions)
                end_loss = compute_loss(end_logits, end_positions)
            else:
                start_positions_list = features["start_position_list"]
                end_positions_list = features["end_position_list"]
                answer_positions_mask = features["position_mask"]

                if FLAGS.marginalize:
                    tf.logging.info('Using margmalized loss for training')
                    start_loss = compute_marginalized_loss(
                        start_logits,
                        start_positions_list,
                        answer_positions_mask
                    )
                    end_loss = compute_marginalized_loss(
                        end_logits,
                        end_positions_list,
                        answer_positions_mask
                    )
                elif FLAGS.posterior_distillation:
                    tf.logging.info('Using PD loss for training')
                    start_loss = compute_pd_loss(
                        start_logits,
                        start_positions_list,
                        answer_positions_mask
                    )
                    end_loss = compute_pd_loss(
                        end_logits,
                        end_positions_list,
                        answer_positions_mask
                    )
                else:
                    raise ValueError('Neither marginalization or pd is used!')

            total_loss = (start_loss + end_loss) / 2.0

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["start_position_list"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )
        name_to_features["end_position_list"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )
        name_to_features["position_mask"] = tf.FixedLenFeature(
            [FLAGS.max_short_answers], tf.int64
            )

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def doc_normalization(results, unique_id_to_qid):
    """Normalizes each RawResult using the qid."""

    qid_to_start_logits = collections.defaultdict(list)
    qid_to_end_logits = collections.defaultdict(list)

    for result in results:
        qid = unique_id_to_qid.get(result.unique_id, None)
        if not qid:
            raise ValueError("Unknown qid for unique_id {0}".format(
                result.unique_id))
        qid_to_start_logits[qid].extend(result.start_logits)
        qid_to_end_logits[qid].extend(result.end_logits)

    # Normalizes the scores.
    qid_to_doc_start_score = collections.defaultdict(float)
    qid_to_doc_end_score = collections.defaultdict(float)

    for qid in qid_to_start_logits:
        start_logits = qid_to_start_logits[qid]
        end_logits = qid_to_end_logits[qid]

        qid_to_doc_start_score[qid] = logsumexp(start_logits)
        qid_to_doc_end_score[qid] = logsumexp(end_logits)

    def normalize(scores, normalization_score):
        return [score - normalization_score for score in scores]

    def normalized_result_generator(result_list):
        """The generator function for normalizing the results."""
        for result in result_list:
            qid = unique_id_to_qid[result.unique_id]
            doc_start_score = qid_to_doc_start_score[qid]
            doc_end_score = qid_to_doc_end_score[qid]
            yield RawResult(
                unique_id=result.unique_id,
                start_logits=normalize(result.start_logits, doc_start_score),
                end_logits=normalize(result.end_logits, doc_end_score)
            )

    # Iterates for the second time to apply the document level normalization.
    new_results = [norm_result
                   for norm_result in normalized_result_generator(results)]

    return new_results


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      prob_transform_func):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if FLAGS.version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = prob_transform_func(total_scores)

        if not best_non_null_entry:
            tf.logging.info("No non-null guess")
            best_non_null_entry = _NbestPrediction(
                text="empty", start_logit=0.0, end_logit=0.0
                )

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _compute_exp(scores):
    """Computes expoent score over normalized logits."""
    if not scores:
        return []

    return [math.exp(x) for x in scores]


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature(
                [feature.start_position])
            features["end_positions"] = create_int_feature(
                [feature.end_position])
            features["start_position_list"] = create_int_feature(
                feature.start_position_list)
            features["end_position_list"] = create_int_feature(
                feature.end_position_list)
            features["position_mask"] = create_int_feature(
                feature.position_mask)

            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train and not FLAGS.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")

    if FLAGS.do_predict and not FLAGS.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

    if FLAGS.max_num_answer_strings < FLAGS.max_short_answers:
        raise ValueError(
            "The max_num_answer_strings (%d) must be bigger than "
            "max_short_answers (%d)" % (FLAGS.max_num_answer_strings,
                                        FLAGS.max_short_answers)
        )

    if FLAGS.local_obj_alpha > 0.0:
        tf.logging.info("Using local_obj_alpha=%f" % FLAGS.local_obj_alpha)


class DocQAModel(object):
    """Document QA model."""

    def __init__(self, bert_config, mode):
        # Makes those variables as local variables.
        self.max_seq_length = FLAGS.max_seq_length
        self.max_num_answers = FLAGS.max_short_answers
        self.max_num_answer_strings = FLAGS.max_num_answer_strings

        self._inputs, self._outputs = self.build_model(mode, bert_config)

        self._fetch_var_names = []
        self._fetch_var_names.append('loss_to_opt')
        if mode != 'TRAIN':
            self._fetch_var_names.append('start_logits', 'end_logits')

    def check_fetch_var(self):
        """Checks whether all variables to fetch are in the output dict."""
        # Checks whether required variables are in the outputs_.
        for var_name in self._fetch_var_names:
            if var_name not in self._outputs:
                raise ValueError(
                    '{0} is not in the output list'.format(var_name))

    def build_model(self, mode, bert_config, use_one_hot_embeddings=False):
        """Builds the model based on BERT."""
        input_ids = tf.placeholder(
            tf.int64, name='input_ids', shape=[None, self.max_seq_length]
        )
        input_mask = tf.placeholder(
            tf.int64, name='input_mask', shape=[None, self.max_seq_length]
        )
        segment_ids = tf.placeholder(
            tf.int64, name="segment_ids", shape=[None, self.max_seq_length]
        )

        is_training = (mode == 'TRAIN')

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        inputs = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

        return inputs, outputs

    def build_loss(self):
        """Builds loss variables."""

        start_logits = self._outputs["start_logits"]
        end_logits = self._outputs["end_logits"]

        input_shape = modeling.get_shape_list(self._inputs['input_ids'],
                                              expected_rank=2)
        batch_size, seq_length = input_shape[0], input_shape[1]

        def one_hot_answer_positions(position_list, position_mask, depth):
            position_tensor = tf.one_hot(
                position_list, depth, dtype=tf.float32)
            position_masked = position_tensor * tf.cast(
                tf.expand_dims(position_mask, -1), dtype=tf.float32
            )
            onehot_positions = tf.reduce_max(position_masked, axis=1)
            return onehot_positions

        def compute_masked_logits(logits, position_list, answer_masks):
            position_tensor = one_hot_answer_positions(
                position_list, answer_masks, seq_length
            )
            return logits + tf.log(position_tensor)

        def compute_marginalized_loss(logits, masked_logits, axis=-1):
            z_positive = tf.reduce_logsumexp(masked_logits, axis=axis)
            z_all = tf.reduce_logsumexp(logits, axis=axis)
            return tf.reduce_sum(z_all - z_positive)

        def compute_pd_loss(logits, masked_logits, axis=-1):
            masked_logits *= FLAGS.teacher_temperature
            fixed_teacher_dist = tf.stop_gradient(
                tf.nn.softmax(masked_logits, axis=axis)
            )

            fixed_teacher_dist = tf.check_numerics(
                fixed_teacher_dist, "fixed_teacher_dist has numeric problem"
            )

            if FLAGS.pd_loss == "sqerr":
                tf.logging.info("Using squared hellinger distance!")

                model_dist = tf.nn.softmax(logits, axis=axis)
                return tf.reduce_sum(tf.squared_difference(
                    tf.sqrt(fixed_teacher_dist),
                    tf.sqrt(model_dist)
                ))
            else:
                tf.logging.info("Using KL-divergence!")
                return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=fixed_teacher_dist,
                    logits=logits,
                    dim=axis
                ))

        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_mean(
                tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            return loss

        if FLAGS.train_first_answer:
            raise ValueError("Not support yet!")
            start_positions = tf.placeholder(tf.int64, name="start_positions",
                                             shape=[None])
            end_positions = tf.placeholder(tf.int64, name="end_positions",
                                           shape=[None])

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            self._inputs['start_positions'] = start_positions
            self._inputs['end_positions'] = end_positions
        else:
            # Builds input placeholder variables.
            start_positions_list = tf.placeholder(
                tf.int32, name="start_position_list",
                shape=[None, self.max_num_answers]
            )
            end_positions_list = tf.placeholder(
                tf.int32, name="end_position_list",
                shape=[None, self.max_num_answers]
            )
            answer_positions_mask = tf.placeholder(
                tf.int64, name="answer_positions_mask",
                shape=[None, self.max_num_answers]
            )

            answer_index_list = tf.placeholder(
                tf.int32, name="answer_index_list",
                shape=[None, self.max_num_answers]
            )

            # Only keeps no-null answer positions.
            doc_answer_positions_mask = tf.where(
                answer_index_list > 0,
                answer_positions_mask,
                tf.zeros_like(answer_positions_mask)
            )

            doc_answer_positions_mask = tf.stop_gradient(
                doc_answer_positions_mask)

            # Builds the input-variable map.
            self._inputs['start_positions_list'] = start_positions_list
            self._inputs['end_positions_list'] = end_positions_list
            self._inputs['answer_positions_mask'] = answer_positions_mask
            self._inputs['answer_index_list'] = answer_index_list

            # This masked logits do not contain the null answer positions.
            doc_masked_start_logits = compute_masked_logits(
                start_logits, start_positions_list, doc_answer_positions_mask
            )
            doc_masked_end_logits = compute_masked_logits(
                end_logits, end_positions_list, doc_answer_positions_mask
            )

            # This masked logits contain the null answer positions.
            par_masked_start_logits = compute_masked_logits(
                start_logits, start_positions_list, answer_positions_mask
            )
            par_masked_end_logits = compute_masked_logits(
                end_logits, end_positions_list, answer_positions_mask
            )

            def compute_logprob(logits, axis=-1):
                return logits - tf.reduce_logsumexp(logits, axis=axis)

            # Groups answer span probilities by the answer string.
            def group_answer_span_prob(ans_span_probs, group_ids):
                """Sums all answer span probilities from the same group."""
                delta = self.max_num_answer_strings + 1

                group_ans_probs = tf.math.unsorted_segment_sum(
                    tf.reshape(ans_span_probs, [-1]),
                    tf.reshape(group_ids, [-1]),
                    delta
                )

                return tf.reshape(
                    group_ans_probs, [1, delta]
                )

            def compute_answer_string_probs(
                    masked_start_logits, masked_end_logits, start_pos_list,
                    end_pos_list, ans_str_indices, axis=None):
                """Computes the answer string prob given the position logits."""
                start_log_probs = compute_logprob(
                    masked_start_logits, axis=axis
                )
                end_log_probs = compute_logprob(
                    masked_end_logits, axis=axis
                )

                # Note: tf.int32 is required for tf.batch_gather.
                ans_span_start_log_probs = tf.batch_gather(
                    start_log_probs, start_pos_list
                )
                ans_span_end_log_probs = tf.batch_gather(
                    end_log_probs, end_pos_list
                )

                span_probs = tf.exp(ans_span_start_log_probs +
                                    ans_span_end_log_probs)

                # Marginalizes the answer span probabilities.
                # The result probabilities is over answer strings unique in the
                # document.
                str_probs = group_answer_span_prob(
                    span_probs, ans_str_indices
                )

                return str_probs, span_probs

            # If True, performs label cleaning.
            if FLAGS.label_cleaning:
                # Computes the answer string probabilities.
                ans_str_probs, ans_span_probs = compute_answer_string_probs(
                    doc_masked_start_logits,
                    doc_masked_end_logits,
                    start_positions_list,
                    end_positions_list,
                    answer_index_list,
                    axis=None
                )

                # For document-level normalization,
                # ans_str_probs is a [1, max_num_answer_strings]-shaped Tensor.
                best_answer_str_indices = tf.tile(tf.expand_dims(tf.argmax(
                    ans_str_probs, axis=-1, output_type=tf.int32
                ), axis=-1), [batch_size, self.max_num_answers])

                best_answer_position_masks = tf.cast(tf.equal(
                    best_answer_str_indices, answer_index_list), tf.int64)

                # Only keeps those positions that are valid answer positions.
                # In other words, null-answer spans for each paragraph is masked
                # out.
                best_answer_position_masks *= doc_answer_positions_mask

                best_answer_position_masks = tf.stop_gradient(
                    best_answer_position_masks)

                # Recomputes the masked logits based on the cleaned labels.
                cleaned_start_logits = compute_masked_logits(
                    start_logits,
                    start_positions_list,
                    best_answer_position_masks
                )

                cleaned_end_logits = compute_masked_logits(
                    end_logits,
                    end_positions_list,
                    best_answer_position_masks
                )

            start_loss, end_loss = 0.0, 0.0
            def single_doc_logits(logits):
                return tf.reshape(logits, [1, -1])

            if FLAGS.marginalize:
                tf.logging.info('Using margmalized loss for training')
                if FLAGS.label_cleaning:
                    tf.logging.info('Performing label cleaning for training')
                    start_loss += compute_marginalized_loss(
                        single_doc_logits(start_logits),
                        single_doc_logits(cleaned_start_logits)
                    )
                    end_loss += compute_marginalized_loss(
                        single_doc_logits(end_logits),
                        single_doc_logits(cleaned_end_logits)
                    )

                if FLAGS.local_obj_alpha > 0.0:
                    start_loss += compute_marginalized_loss(
                        start_logits,
                        par_masked_start_logits
                    ) * FLAGS.local_obj_alpha
                    end_loss += compute_marginalized_loss(
                        end_logits,
                        par_masked_end_logits
                    ) * FLAGS.local_obj_alpha

            elif FLAGS.posterior_distillation:
                tf.logging.info('Using PD loss for training')

                if FLAGS.label_cleaning:
                    tf.logging.info('Performing label cleaning for training')
                    start_loss += compute_pd_loss(
                        single_doc_logits(start_logits),
                        single_doc_logits(cleaned_start_logits)
                    )
                    end_loss += compute_pd_loss(
                        single_doc_logits(end_logits),
                        single_doc_logits(cleaned_end_logits)
                    )

                if FLAGS.local_obj_alpha > 0.0:
                    start_loss += FLAGS.local_obj_alpha * compute_pd_loss(
                        start_logits,
                        par_masked_start_logits
                    )
                    end_loss += FLAGS.local_obj_alpha * compute_pd_loss(
                        end_logits,
                        par_masked_end_logits
                    )

            else:
                raise ValueError('Neither marginalization or pd is used!')

        total_loss = (start_loss + end_loss) / 2.0

        self._outputs['loss_to_opt'] = total_loss

    def build_opt_op(self, learning_rate, num_train_steps, num_warmup_steps,
                     use_tpu=False):
        """Builds optimization operator for the model."""
        loss_to_opt = self._outputs['loss_to_opt']

        return optimization.create_optimizer(
            loss_to_opt, learning_rate, num_train_steps, num_warmup_steps,
            use_tpu
        )

    def _run_model(self, session, feed_dict, opt_op):
        """Performans a forward and backward pass of the model."""

        fetches = [self._outputs[var_name]
                   for var_name in self._fetch_var_names]
        fetches.append(opt_op)

        all_outputs = session.run(fetches, feed_dict)

        fetched_var_dict = dict([
            (var_name, all_outputs[idx])
            for idx, var_name in enumerate(self._fetch_var_names)
        ])

        return fetched_var_dict

    def _build_feed_dict(self, inputs_dict):
        """Builds feed dict for inputs."""
        feed_dict_list = []
        for input_name, input_var in self._inputs.items():
            if input_name not in inputs_dict:
                raise ValueError('Missing input_name: {0}'.format(input_name))
            feed_dict_list.append((input_var, inputs_dict[input_name]))
        return dict(feed_dict_list)

    def one_step(self, session, inputs_dict, opt_op):
        """Trains, evaluates, or infers the model with one batch of data."""
        feed_dict = self._build_feed_dict(inputs_dict)
        fetched_dict = self._run_model(session, feed_dict, opt_op)

        return fetched_dict

    def initialize_from_checkpoint(self, init_checkpoint):
        """Initializes model variables from init_checkpoint."""
        variables_to_restore = tf.trainable_variables()

        if init_checkpoint:
            tf.logging.info(
                "Initializing the model from {0}".format(init_checkpoint))
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 variables_to_restore, init_checkpoint
             )

            # TODO(chenghao): Currently, no TPU is supported.
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in variables_to_restore:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


class InputFeatureContainer(object):
    """Data Container for InputFeature."""

    def __init__(self, feature_filename, num_sample, batch_size, shuffle_data,
                 rand_seed, chunk_size=1000):

        np.random.seed(rand_seed)
        self.shuffle_data = shuffle_data
        self.input_feature_list = []
        self.batch_size = batch_size
        self.feature_filename = feature_filename
        self.chunk_size = chunk_size
        self.num_sample = num_sample
        self.processed_sample_cnt = self.num_sample
        self.rng = random.Random(rand_seed)

        if not tf.gfile.Exists(feature_filename):
            raise ValueError(
                'Feature file {0} doesn not exist'.format(feature_filename))

    def start_epoch(self, batch_size=None):
        """Prepares for a new epoch."""
        tf.logging.info("The data reader has processed {0} features".format(
            self.processed_sample_cnt))

        tf.logging.info("Starts a new epoch!")

        # This might be too costly.
        # if self.shuffle_data:
        #     tf.logging.info("Shuffles the file globally.")
        #     with tf.gfile.Open(self.feature_filename, mode='r') as fin:
        #         lines = [line for line in fin]

        #     self.rng.shuffle(lines)
        #     with tf.gfile.GFile(self.feature_filename, mode='w') as fout:
        #         for line in lines:
        #             fout.write(line)

        #     del lines

        #     tf.logging.info(
        #         "Shuffle data of chunk_size {0}".format(self.chunk_size))

        self.processed_sample_cnt = 0

        tf.logging.info(
            'The new epoch will have batch_size: {0}'.format(self.batch_size))

    def extract_batches(self, mode):
        """Iterates over the dataset."""

        is_training = (mode == 'TRAIN')
        chunk_size = self.chunk_size
        batch_size = self.batch_size

        def empty_func(_):
            pass

        def group_key_func(group_size):
            """Returns the lambda function for grouping key."""
            return lambda _, fc=itertools.count(): fc.next() // group_size

        shuffler = empty_func
        if self.shuffle_data:
            shuffler = self.rng.shuffle

        def is_valid_answer_doc(answer_indices):
            """Checks whether the document has at least one valid answer."""
            return any(map(lambda x: x > 0, answer_indices))

        with tf.gfile.Open(self.feature_filename, mode='r') as fin:
            for _, file_chunk in itertools.groupby(
                    fin, key=group_key_func(chunk_size)):

                feature_list = [
                    DocInputFeatures.load_from_json(feature_str)
                    for feature_str in file_chunk
                ]

                shuffler(feature_list)

                # Given this is super large, we can only train one doc per step.
                for document in feature_list:
                    has_answer = False if is_training else True

                    while not has_answer:
                        unique_ids = []
                        input_ids = []
                        input_mask = []
                        segment_ids = []

                        start_positions_list = []
                        end_positions_list = []
                        answer_positions_mask = []

                        answer_index_list = []

                        num_sample = 0

                        shuffler(document.feature_list)

                        for feature in itertools.islice(
                                document.feature_list,
                                0, FLAGS.max_num_doc_feature):

                            unique_ids.append(feature.unique_id)
                            input_ids.append(feature.input_ids)
                            input_mask.append(feature.input_mask)
                            segment_ids.append(feature.segment_ids)

                            if is_training:
                                start_positions_list.append(
                                    feature.start_position_list)
                                end_positions_list.append(
                                    feature.end_position_list)
                                answer_positions_mask.append(
                                    feature.position_mask)
                                answer_index_list.append(
                                    feature.answer_index_list)

                                # Answer index 0 is reserved for null.
                                if is_valid_answer_doc(
                                        feature.answer_index_list):
                                    has_answer = True

                            num_sample += 1

                    self.processed_sample_cnt += num_sample

                    if is_training and not has_answer:
                        raise ValueError("No answer document!")

                    feature_dict = {
                        'unique_ids': unique_ids,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids,
                        'start_positions_list': start_positions_list,
                        'end_positions_list': end_positions_list,
                        'answer_positions_mask': answer_positions_mask,
                        'answer_index_list': answer_index_list,
                        'num_sample': num_sample,
                        'num_unique_ans_str': feature.num_answer,
                    }

                    yield feature_dict


def run_epoch(model, session, data_container, learning_rate, opt_op, mode,
              model_saver=None, verbose=True, eval_func=None):
    """Runs one epoch over the data."""
    start_time = time.time()
    iter_count = 1

    process_sample_cnt = 0

    # Starts a new epoch.
    data_container.start_epoch()

    total_loss = 0.0

    for feature_dict in data_container.extract_batches(mode):

        fetched_dict = model.one_step(session, feature_dict, opt_op)
        total_loss += fetched_dict['loss_to_opt']

        iter_count += 1

        process_sample_cnt += feature_dict['num_sample']

        if (iter_count) % 1000 == 0:
            if verbose:
                tf.logging.info(
                    'iter {:d}:, {:.3f} examples per second'.format(
                        iter_count,
                        process_sample_cnt / (time.time() - start_time)
                    )
                )
                tf.logging.info(
                    'loss {:.3f}'.format(fetched_dict['loss_to_opt'])
                )

            if model_saver:
                global_step = session.run(tf.train.get_global_step())
                model_saver.save(
                    session,
                    os.path.join(FLAGS.output_dir, 'model.ckpt'),
                    global_step=global_step
                )

    tf.logging.info(
        'time for one epoch: {:.3f} secs'.format(time.time() - start_time)
    )
    tf.logging.info('iters over {0} num of samples'.format(process_sample_cnt))

    eval_metric = {'total_loss': total_loss}
    output_dict = {}

    return total_loss, eval_metric, output_dict


def train_model(train_feature_filename, num_sample, bert_config, learning_rate,
                num_train_steps, num_warmup_steps, batch_size, shuffle_data,
                init_checkpoint, rand_seed=12345, chunk_size=1000):
    """ Training wrapper function."""
    if FLAGS.device == 'cpu':
        session_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=FLAGS.num_cpus,
            inter_op_parallelism_threads=FLAGS.num_cpus,
            allow_soft_placement=True
        )
    else:
        session_config = tf.ConfigProto(
            intra_op_parallelism_threads=FLAGS.num_cpus,
            inter_op_parallelism_threads=FLAGS.num_cpus,
            allow_soft_placement=True
        )
        session_config.gpu_options.allow_growth = True

    # A chunk of the data would sit in memory for the whole training process.
    train_data_container = InputFeatureContainer(
        train_feature_filename, num_sample, batch_size, shuffle_data,
        rand_seed, chunk_size=chunk_size
    )
    valid_data_container = None

    # if not os.path.exists(train_config.save_model_dir):
    if not os.path.exists(FLAGS.output_dir):
        raise ValueError(
            'output_dir ({0}) does not exist!'.format(
                FLAGS.output_dir)
        )

    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        tf.set_random_seed(rand_seed)
        np.random.seed(rand_seed)
        # initializer = _build_initializer(FLAGS.initializer)

        model = DocQAModel(bert_config, 'TRAIN')
        model_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)

        # This is needed for both TRAIN and EVAL.
        model.build_loss()
        model.check_fetch_var()

        # This operation is only needed for TRAIN phase.
        opt_op = model.build_opt_op(learning_rate, num_train_steps,
                                    num_warmup_steps)

        # Loads pretrain model parameters if specified.
        if init_checkpoint:
            model.initialize_from_checkpoint(init_checkpoint)

        session.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        for it in xrange(int(FLAGS.num_train_epochs)):
            tf.logging.info('Train Iter {0}'.format(it))

            _, train_metric, _ = run_epoch(
                model, session, train_data_container, None,
                opt_op, 'TRAIN', eval_func=None, model_saver=model_saver
            )

            tf.logging.info('\n'.join([
                'train {}: {:.3f}'.format(metric_name, metric_val)
                for metric_name, metric_val in train_metric.items()
            ]))

            tf.logging.info('Saves the current model.')
            global_step = session.run(tf.train.get_global_step())
            model_saver.save(
                session,
                os.path.join(FLAGS.output_dir, 'model.ckpt'),
                global_step=global_step
            )

        tf.logging.info('Saves the final model.')
        global_step = session.run(tf.train.get_global_step())
        model_saver.save(
            session,
            os.path.join(FLAGS.output_dir, 'model.ckpt'),
            global_step=global_step
        )
        tf.logging.info('Training model done!')

    return True


def prediction_generator(estimator, predict_input_fn):
    """Given the input fn and estimator, yields one result."""
    for cnt, result in enumerate(estimator.predict(
            predict_input_fn, yield_single_examples=True)):
        if cnt % 1000 == 0:
            tf.logging.info("Processing example: %d" % cnt)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        yield RawResult(unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=10,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:

        # This is a new adjustion to the training approach.
        train_filename = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not tf.gfile.Exists(train_filename):
            tf.logging.info('Processes examples')
            train_examples = read_doc_squad_examples_from_generator(
                input_file=FLAGS.train_file, is_training=True)

            # Pre-shuffle the input to avoid having to make a very large shuffle
            # buffer in in the `input_fn`.
            rng = random.Random(12345)
            rng.shuffle(train_examples)

            with tf.gfile.GFile(train_filename, mode='w') as fout:
                tf.logging.info('Converting Doc Example into Doc InputFeatures')
                num_train_doc_features = convert_doc_examples_to_feature_list(
                    doc_examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=FLAGS.max_seq_length,
                    doc_stride=FLAGS.doc_stride,
                    max_query_length=FLAGS.max_query_length,
                    output_fn=fout.write,
                    is_training=True
                )

            num_train_features = len(train_examples)
            del train_examples
        else:
            tf.logging.info(
                "Reads InputFeatures directly from {0}".format(train_filename))
            num_train_doc_features = -1
            num_train_features = 0
            with tf.gfile.Open(train_filename, mode='r') as fin:
                for _ in fin:
                    num_train_features += 1

        num_train_steps = int(
            num_train_features /
            FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num doc examples = %d", num_train_features)
        tf.logging.info("  Num split examples = %d", num_train_doc_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_model(train_feature_filename=train_filename,
                    num_sample=num_train_features,
                    bert_config=bert_config,
                    learning_rate=FLAGS.learning_rate,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                    batch_size=FLAGS.train_batch_size,
                    shuffle_data=FLAGS.shuffle_data,
                    init_checkpoint=FLAGS.init_checkpoint,
                    rand_seed=12345)

    # ======================================================================

    if FLAGS.do_predict:

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

        # If TPU is not available, this will fall back to normal Estimator on
        # CPU or GPU.
        predict_batch_size = FLAGS.predict_batch_size

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=predict_batch_size
        )

        eval_examples = read_squad_examples_from_generator(
            input_file=FLAGS.predict_file, is_training=False)
        eval_record_filename = os.path.join(FLAGS.output_dir, "eval.tf_record")
        eval_feature_filename = os.path.join(FLAGS.output_dir, "eval.features")

        unique_id_to_qid = collections.defaultdict()
        if not tf.gfile.Exists(eval_record_filename):
            tf.logging.info("Converting examples into records and features.")

            eval_writer = FeatureWriter(
                filename=eval_record_filename,
                is_training=False)

            eval_features = []

            def append_feature(feature):
                eval_features.append(feature)
                eval_writer.process_feature(feature)

            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=False,
                output_fn=append_feature,
                unique_id_to_qid=unique_id_to_qid
            )
            eval_writer.close()
            with tf.gfile.GFile(eval_feature_filename, mode="w") as fout:
                for feature in eval_features:
                    fout.write(feature.to_json())
                    fout.write('\n')

        else:
            tf.logging.info("Reusing converted records and features.")

            with tf.gfile.Open(eval_feature_filename, mode="r") as fin:
                eval_features = [
                    InputFeatures.load_from_json(line)
                    for line in fin
                ]

            for feature in eval_features:
                if feature.unique_id is None:
                    raise ValueError("InputFeatures should have unique_id!")
                if feature.qid is None:
                    raise ValueError("InputFeatures should have qid!")

                unique_id_to_qid[feature.unique_id] = feature.qid

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_record_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        # all_results = []
        # for result in estimator.predict(
        #         predict_input_fn, yield_single_examples=True):
        #     if len(all_results) % 1000 == 0:
        #         tf.logging.info("Processing example: %d" % (len(all_results)))
        #     unique_id = int(result["unique_ids"])
        #     start_logits = [float(x) for x in result["start_logits"].flat]
        #     end_logits = [float(x) for x in result["end_logits"].flat]
        #     all_results.append(
        #         RawResult(
        #             unique_id=unique_id,
        #             start_logits=start_logits,
        #             end_logits=end_logits))

        all_results = [
            raw_result
            for raw_result in prediction_generator(estimator, predict_input_fn)
        ]

        tf.logging.info("Done prediction!")
        del estimator
        del predict_input_fn

        if FLAGS.doc_normalize:
            tf.logging.info("Performs document level normalization.")
            normalized_results = doc_normalization(
                all_results, unique_id_to_qid
            )

            # For document-level normalization, each score is a log-prob.
            prob_trans_func = _compute_exp
        else:
            tf.logging.info("Performs paragraph level normalization.")
            normalized_results = all_results

            # For paragraph-level normalization, each score is a logit.
            prob_trans_func = _compute_softmax

        output_prediction_file = os.path.join(
            FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            FLAGS.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            FLAGS.output_dir, "null_odds.json")

        write_predictions(eval_examples, eval_features, normalized_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file,
                          prob_trans_func)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
