
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
"""Utilities and classes for objects that provide an interestingness score.

The Appraiser class holds a mapping from items (described as strings) to
scores for them, as judged by some precomputed calculation. For example,
the function create_idf_appraiser() goes over all Matterport3D scans and
computes an IDF score for each clean category, treating each panorama as a
document and the categories of the observable objects as the words in the
document. Scores for predefined categories can be boosted or penalized. When,
an Appraiser is created, these are used to create adjusted scores for each
category.

Appraisers can be written to a file in a string representation and read back for
use in later applications that need a measure of how intrinsically interesting
an object is.
"""

import collections
import copy
import math
from typing import Dict, FrozenSet, Sequence, Text

import attr
import tensorflow as tf

_SEPARATOR = '::'


@attr.s
class Appraiser:
  """A scorer of items based on how interesting they are.

  This is basically just a dictionary from strings to floats, which could be
  created any way one likes. However, it adds some options for boosting or
  penalizing some items that are hand selected.

  `raw_scores` are the scores of each item without the hand selected boosts or
  penalities. This is kept around in case one wants direct access to the
  interestingness scores as determined by the data and chosen algorithm.

  `occlusion_threshold` is the float governing how sensitive the object
  observer is to whether and by how much an object is visibly occluded by
  another from the center of the panorama associated with it. It is not needed
  by the Appraiser itself, but it is used when generating the observations, and
  storing it in the Appraiser allows later users of the Appraiser to use the
  same threshold consistently when observing objects from panoramas and
  assessing their interestingness and visual prominence.

  `notable_boost` is a float indicating how much to add to the score of all
  items listed in the `notable_items` set.

  `boring_penality` is a float indicating how much to subtract from the score of
  all items listed in the `boring_items` set.

  `avg_score` is the average of all scores provided in `raw_scores`

  `adjusted_scores` are the modified scores per item based on the boosts and
  penalities for notability and boringness. This is computed up front during
  the creation of the Appraiser object.
  """

  raw_scores: Dict[Text, float] = attr.ib()
  occlusion_threshold: float = attr.ib()
  notable_boost: float = attr.ib()
  boring_penalty: float = attr.ib()
  notable_items: FrozenSet[Text] = attr.ib()
  boring_items: FrozenSet[Text] = attr.ib()
  avg_score: float = attr.ib(init=False)
  adjusted_scores: Dict[Text, float] = attr.ib(init=False)

  def __attrs_post_init__(self):
    """Computes the adjusted scores dictionary."""
    self.avg_score = sum(self.raw_scores.values()) / len(self.raw_scores)

    self.adjusted_scores = copy.deepcopy(self.raw_scores)
    for item in self.notable_items:
      self.adjusted_scores[item] = (
          self.raw_scores.get(item, 0.0) + self.notable_boost)
    for item in self.boring_items:
      self.adjusted_scores[item] = max(
          self.raw_scores.get(item, 0.0) - self.boring_penalty, math.log(1.1))

  def __call__(self, item: Text, multiplier=1.0) -> float:
    """Returns the score, including boosts and penalities)."""
    return self.adjusted_scores.get(item, self.avg_score) * multiplier

  def get_raw_score(self, item: Text) -> float:
    """Returns the raw score of the item without adjustments."""
    return self.raw_scores.get(item, self.avg_score)

  def to_file(self, filename: Text):
    """Writes this Appraiser to text representation on disk."""
    with tf.io.gfile.GFile(filename, 'w') as output:
      output.write(
          f'occlusion_threshold{_SEPARATOR}{self.occlusion_threshold}\n')
      output.write(f'notable_boost{_SEPARATOR}{self.notable_boost}\n')
      output.write(f'boring_penalty{_SEPARATOR}{self.boring_penalty}\n')

      output.write(
          f'Count of notable items{_SEPARATOR}{len(self.notable_items)}\n')
      for item in self.notable_items:
        output.write(f'{item}\n')

      output.write(
          f'Count of boring items{_SEPARATOR}{len(self.boring_items)}\n')
      for item in self.boring_items:
        output.write(f'{item}\n')

      for k, v in self.raw_scores.items():
        output.write(f'{k}{_SEPARATOR}{v}\n')
      output.flush()

  @classmethod
  def from_file(cls, filename: Text):
    """Loads an Appraiser from text representation on disk."""

    with tf.io.gfile.GFile(filename, 'r') as inputfile:
      occlusion_threshold = float(inputfile.next().split(_SEPARATOR)[1])
      notable_boost = float(inputfile.next().split(_SEPARATOR)[1])
      boring_penalty = float(inputfile.next().split(_SEPARATOR)[1])

      num_notable = int(inputfile.next().split(_SEPARATOR)[1])
      notable_items = frozenset(
          [inputfile.next().rstrip('\n') for _ in range(num_notable)])

      num_boring = int(inputfile.next().split(_SEPARATOR)[1])
      boring_items = frozenset(
          [inputfile.next().rstrip('\n') for _ in range(num_boring)])

      scores = {}
      for line in inputfile:
        k, v = line.split(_SEPARATOR)
        scores[k] = float(v)

    return cls(scores, occlusion_threshold, notable_boost, boring_penalty,
               notable_items, boring_items)


# Hand-selected clean categories to boost as extra interesting to refer to.
_DEFAULT_NOTABLE_CATEGORIES = frozenset([
    'bulletin board', 'computer', 'massage table', 'exercise bike',
    'chest_of_drawers', 'shower bench', 'display case', 'massage bed', 'easel',
    'bidet', 'bathroom counter', 'whiteboard', 'car', 'shrubbery',
    'water cooler', 'display cabinet', 'computer desk', 'sofa', 'sofa set',
    'side table', 'swivel chair', 'floor lamp', 'tv_monitor', 'treadmill',
    'gym_equipment', 'clothes dryer', 'dishwasher', 'chest of drawers',
    'office table', 'garage door', 'guitar', 'stairs', 'staircase',
    'pool table', 'radiator', 'lounge chair', 'banister', 'bust', 'sculpture',
    'washing machine', 'office chair', 'desk chair', 'vanity', 'tree',
    'monitor', 'tv stand', 'telephone', 'piano', 'clock', 'kitchen island',
    'wardrobe', 'ceiling fan', 'microwave', 'chandelier', 'faucet', 'stove',
    'oven', 'pew', 'altar', 'washbasin', 'bookshelf', 'kitchen counter',
    'bath cabinet', 'statue', 'archway', 'dining chair', 'end table', 'dresser',
    'ottoman', 'refrigerator', 'countertop', 'dining table', 'fan', 'bathtub',
    'kitchen cabinet', 'counter', 'desk', 'railing', 'vase', 'stool', 'toilet',
    'bench', 'trashcan', 'armchair', 'handrail', 'sofa chair', 'coffee table',
    'fireplace', 'nightstand', 'pillar', 'shelf', 'stair', 'sink', 'bed', 'tv',
    'couch', 'lamp', 'mirror', 'cabinet', 'plant', 'table', 'chair', 'picture'
])

# Categories that are generally less useful to refer to during navigation.
_DEFAULT_BORING_CATEGORIES = frozenset(
    ['ceiling', 'floor', 'wall', 'lighting', 'object', 'objects', 'roof'])


def create_idf_appraiser(documents: Sequence[Sequence[Text]],
                         notable_boost: float, boring_penalty: float,
                         occlusion_threshold: float) -> Appraiser:
  """Creates an Appraiser based on inverse document frequencies for categories.

  Args:
    documents: A sequence of word sequences. The expected use case is one in
      which each "word" is a Matterport3D category that is present based on an
      instance of an object of that category being visible in a panorama. (Thus,
      a panorama is a "document".
    notable_boost: A float indicating how much to add to the score of any
      category indicated in _DEFAULT_NOTABLE_CATEGORIES.
    boring_penalty: A float indicating how much to subtract from any category
      indicated in _DEFAULT_BORING_CATEGORIES.
    occlusion_threshold: The threshold that was used to determine occlusion when
      the objects were observed to create the panorama document. This is not
      used in the score computations, but is needed for consistency in later use
      of the Appraiser.

  Returns:
    An Appraiser with IDF scores for each category.
  """

  num_documents = len(documents)
  if not num_documents:
    raise ValueError('Cannot create Appraiser: no documents provided!')

  item_idf = collections.defaultdict(lambda: math.log(1.1))

  # Compute count of each item appearance in all documents.
  item_doc_freq = collections.defaultdict(int)
  for document in documents:
    for item in set(document):
      item_doc_freq[item] += 1

  # Compute IDF values
  for item, count in item_doc_freq.items():
    # Use so-called probabilistic IDF, but adding 1.1 to ensure >0 values.
    item_idf[item] = math.log(1.1 + (num_documents - count) / count)

  return Appraiser(item_idf, occlusion_threshold, notable_boost, boring_penalty,
                   _DEFAULT_NOTABLE_CATEGORIES, _DEFAULT_BORING_CATEGORIES)
