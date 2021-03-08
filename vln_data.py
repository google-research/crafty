
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
"""Utilities specifically for Room-to-Room items."""

import json
import os
from typing import Sequence, Text

import tensorflow as tf


class RRItem:
  """Base class for representing an RxR/R2R/R4R item."""

  def __init__(self, dict_item, source='R2R'):
    """Populates the R_RItem using dictionary info."""
    self.source: Text = source
    self.distance: float = float(dict_item['distance'])
    self.scan: str = dict_item['scan']
    self.path_id: int = int(dict_item['path_id'])
    self.path: Sequence[str] = dict_item['path']
    self.heading: float = float(dict_item['heading'])
    self.instructions: Sequence[Text] = dict_item['instructions']


class ExtendedRRItem(RRItem):
  """Extending RRItem with new fields from the extended standard R2R data.

  Source: https://arxiv.org/abs/2004.02707.
  This work introduces the augmented R2R dataset where instructions are
  segmented into subinstructions each of which corresponds to a panorama
  in a path (a path = a sequence of panoramas). The additional data fields
  here store the additional annotation. For more details please refer to
  the paper.
  """

  def __init__(self, dict_item, source='R2R'):
    super().__init__(dict_item, source)
    self.new_instructions = dict_item['new_instructions']
    self.chunk_view = dict_item['chunk_view']
    self.pano_matched_object_seqs = dict_item['pano_matched_object_seqs']


def load_split(
    basedir: str,
    split: str,
    source: str = 'R2R',
    extended: bool = False,
) -> Sequence[RRItem]:
  """Loads the requested R2R split."""

  full_path = get_file_path_for_split(basedir, split, source)
  with tf.io.gfile.GFile(full_path) as train_json:
    dataset = json.load(train_json)
  items = []
  for info in dataset:
    items.append(ExtendedRRItem(info) if extended else RRItem(info))
  return items


def get_file_path_for_split(base_dir: str, split: str, source='R2R') -> str:
  filename = f'{source}_{split}.json'
  return os.path.join(base_dir, filename)
