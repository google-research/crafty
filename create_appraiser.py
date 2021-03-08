
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
"""Create an appraiser based on IDF scores of Matterport3D categories."""

from absl import app
from absl import flags

from crafty import appraise
from crafty import mp3d
from crafty import vln_data

flags.DEFINE_integer('notable_boost', 0,
                     'Extra credit for hand curated notable items.')

flags.DEFINE_integer('boring_penalty', 0,
                     'Penalty to apply to hand curated boring items.')

flags.DEFINE_float(
    'occlusion_threshold', -1,
    'The threshold for whether one object is occluded by another. Default (-1)'
    'indicates no occlusion. 0.3 is a good value for occluding objects.')

flags.DEFINE_string('r2r_basedir', '/path/to/matterport_data',
                    'Path to Room-to-Room data.')

flags.DEFINE_string('output_file', None,
                    'Output file to save Riveter information.')

flags.mark_flag_as_required('output_file')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  basedir = FLAGS.r2r_basedir
  mp_data = mp3d.MatterportData(basedir, FLAGS.occlusion_threshold)

  print('Loading R2R training dataset.')
  dataset = vln_data.load_split(basedir, 'train')

  print('Collecting pano objects.')
  all_scans = set([ex.scan for ex in dataset])
  pano_objects = mp_data.get_per_pano_objects(all_scans)

  print('Calculating IDF scores.')
  appraiser = appraise.create_idf_appraiser(pano_objects, FLAGS.notable_boost,
                                            FLAGS.boring_penalty,
                                            FLAGS.occlusion_threshold)

  print('Saving appraiser.')
  appraiser.to_file(FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
