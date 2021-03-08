
 # Copyright 2021 Google LLC
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #      http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
# Lint as: python3
r"""Create hand-crafted instructions using path directions and salient objects.

This version uses simple templates to give directions and note objects in the
instructions, e.g. 'turn left. go to the couch' and so on.

Example usage:
python3 \
  ~/crafty/create_instructions.py -- \
  --hmm_type='path_specific' \
  --instruction_type='full' \
  --magnitude_normalization='normal' \
  --num_instructions=1 \
  --path_input_dir='/path/to/source_json.json' \
  --dataset='R2R' \
  --file_identifier='val_seen' \
  --output_file='/path/to/save/location/file_prefix'
"""

import copy
import json
import os
from typing import List

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from crafty import appraise
from crafty import guide
from crafty import mp3d
from crafty import observe
from crafty import talk
from crafty import vln_data
from crafty import walk
import numpy as np
import tensorflow as tf

flags.DEFINE_bool('given_landmarks', False,
                  'Whether to use pre-selected landmark objects.')

flags.DEFINE_enum(
    'hmm_type', 'path_specific', ['path_specific', 'hard_em', 'random'],
    'The HMM type. Currently Path-specific, Hard EM, and Random are available.')

flags.DEFINE_enum(
    'instruction_type', 'full',
    ['full', 'object_only', 'direction_only', 'mask_object', 'mask_direction'],
    'Toggle for full/object-only/direction-only instructions.')

flags.DEFINE_enum(
    'magnitude_normalization', 'normal', ['gamma', 'normal'],
    'Distribution type for calculating probability of magnitude for Observer.')

flags.DEFINE_integer('num_instructions', 1,
                     'The number of instructions to generate per path.')

flags.DEFINE_string('mp3d_dir', '/path/to/matterport_data/',
                    'Path to Room-to-Room scan data.')

flags.DEFINE_string('path_input_dir', None, 'Path to Room-to-Room JSON data.')

flags.DEFINE_enum('dataset', None, ['R2R', 'R4R', 'RxR'], 'Data source.')

flags.DEFINE_string(
    'file_identifier', None,
    'Source JSON file identifier for Crafty instruction creation.')

flags.DEFINE_string('output_file', None,
                    'Output file to save generated instructions.')

flags.DEFINE_string(
    'appraiser_file',
    '~/crafty/google/crafty.object_idfs.r2r_train.txt',
    'File to read appraiser information from.')

# Multi-job-specific config:
# The same full training file for all partitions in EM training.
# This is required in multi-job training.
flags.DEFINE_string(
    'full_train_file_path', None,
    'Path to full training file, for EM training covering all partitions.')

flags.mark_flags_as_required(
    ['path_input_dir', 'dataset', 'file_identifier', 'output_file'])

FLAGS = flags.FLAGS

_CRAFTY_VERSION = 'v0.6a'


class RunCrafty(beam.DoFn):
  """Runs Crafty pipeline to generate instructions and save."""

  def __init__(self):
    self._output_file = FLAGS.output_file

  def setup(self):
    super().setup()
    self._mp_data = mp3d.MatterportData(FLAGS.mp3d_dir)
    self._appraiser = appraise.Appraiser.from_file(FLAGS.appraiser_file)
    # Toggles observer type by HMM.
    if FLAGS.hmm_type == 'path_specific':
      observer = observe.PathSpecificObserver(self._mp_data, self._appraiser,
                                              FLAGS.magnitude_normalization, 3,
                                              1.5, 10, 0.1)
    elif FLAGS.hmm_type == 'hard_em':
      observer = observe.HardEMObserver(self._mp_data, self._appraiser,
                                        FLAGS.magnitude_normalization, 3, 1.5,
                                        10, 0.1)
    elif FLAGS.hmm_type == 'random':
      observer = observe.RandomSampleObserver(self._mp_data, self._appraiser,
                                              FLAGS.magnitude_normalization, 3,
                                              1.5, 10, 0.1)
    # Toggles talker type.
    if FLAGS.instruction_type == 'object_only':
      talker = talk.ObjectOnlyTalker(merge_same_object_steps=True)
    elif FLAGS.instruction_type == 'direction_only':
      talker = talk.DirectionOnlyTalker(merge_same_object_steps=True)
    elif FLAGS.instruction_type == 'mask_object':
      talker = talk.TemplateTalker(
          merge_same_object_steps=True, mask_type='object')
    elif FLAGS.instruction_type == 'mask_direction':
      talker = talk.TemplateTalker(
          merge_same_object_steps=True, mask_type='direction')
    else:
      talker = talk.TemplateTalker(merge_same_object_steps=True, mask_type=None)
    self._crafty = guide.Guide(walk.Walker(), talker, observer)
    if FLAGS.hmm_type == 'hard_em':
      if FLAGS.full_train_file_path is None:
        train_data_path = vln_data.get_file_path_for_split(
            FLAGS.path_input_dir, FLAGS.file_identifier)
      else:
        train_data_path = FLAGS.full_train_file_path
      guide.initialize_observer_with_hard_em(self._crafty, train_data_path,
                                             self._mp_data)

  def process(self, data_batch, output_file, batch_index):

    scan_data_index = {}
    output_path = output_file + '_' + str(batch_index) + '.json'
    with tf.io.gfile.GFile(output_path, 'w') as output_json:
      for data_item in data_batch:
        print(data_item.path_id, ' ', end='', flush=True)
        try:
          scan_data = scan_data_index[data_item.scan]
        except KeyError:
          scan_data = self._mp_data.get_scan_data(data_item.scan)
          scan_data_index[data_item.scan] = scan_data

        instructions = []
        for _ in range(FLAGS.num_instructions):
          instructions.append(self._crafty(data_item, scan_data))

        crafty_item = copy.copy(data_item)
        crafty_item.source = f'Crafty {_CRAFTY_VERSION}'
        crafty_item.instructions = instructions

        json.dump(crafty_item, output_json, default=lambda o: o.__dict__)
        output_json.write('\n')
        output_json.flush()


def partition(item_list, num_parts):
  """Partition a list of data items into `num_parts` partitions."""
  return [list(array) for array in np.array_split(item_list, num_parts)]


def main(unused_argv: List[str]) -> None:
  del unused_argv

  run_crafty = RunCrafty()

  def pipeline(root):
    print('Loading designated partition of RxR/R2R/R4R dataset.')
    extended = True if FLAGS.given_landmarks else False
    input_data = vln_data.load_split(FLAGS.path_input_dir,
                                     FLAGS.file_identifier, FLAGS.dataset,
                                     extended)
    logging.info('Starting Beam pipeline.')
    for batch_index, input_data_batch in enumerate(
        partition(input_data, num_parts=10)):
      _ = (
          root
          | f'create_input_{batch_index}' >> beam.Create([input_data_batch])
          | f'do_fn_{batch_index}' >> beam.ParDo(run_crafty, FLAGS.output_file,
                                                 batch_index))

  pipeline_options = beam.options.pipeline_options.PipelineOptions()
  pipeline_options.view_as(beam.options.pipeline_options.DirectOptions
                          ).direct_num_workers = os.cpu_count()
  with beam.Pipeline(options=pipeline_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
