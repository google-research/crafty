
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
"""Helper functions for craftgen."""

import abc
import collections
import itertools
import json
from typing import Dict, Sequence

from crafty import data
import scipy.stats
import tensorflow as tf


class MagnitudeNormalizer(metaclass=abc.ABCMeta):
  """Base class for magnitude normalizer."""

  def __init__(self, scale: float):
    """Initializer.

    Args:
      scale: scale parameter for a given distribution (e.g. std for Gaussian).
    """
    assert scale > 0.0
    self.scale = scale

  @abc.abstractmethod
  def __call__(self, magnitude: float) -> float:
    """Calculates probability of a given magnitude for a given distribution."""
    return

  @abc.abstractmethod
  def cdf(self, magnitude: float) -> float:
    """Calculates the cumulative probability for a given magnitude."""
    return

  @classmethod
  def create(cls, magnitude_normalization: str, loc: float, scale: float):
    if magnitude_normalization == 'normal':
      return GaussianMagnitudeNormalizer(loc, scale)
    elif magnitude_normalization.startswith('gamma'):
      return GammaMagnitudeNormalizer(loc, scale)
    else:
      raise ValueError('Available types are `gamma`, `gamma` and `normal`'
                       ', but got %s.' % magnitude_normalization)


class GaussianMagnitudeNormalizer(MagnitudeNormalizer):
  """Normalizer based on Normal Distribution."""

  def __init__(self, mean: float, std: float):
    """Initializer.

    Args:
      mean: mean param for Gaussian.
      std: std param for Gaussian.
    """
    super().__init__(scale=std)
    self.normal_dist = scipy.stats.norm(mean, self.scale)

  def __call__(self, magnitude: float) -> float:
    """Calculates probability of a given magnitude for a normal distribution."""
    # Get cumulative probability from -infinity to magnitude.
    assert magnitude >= 0.0
    return 2 * (1 - self.cdf(magnitude))

  def cdf(self, magnitude: float) -> float:
    """Calculates the cumulative probability for a given magnitude."""
    assert magnitude >= 0.0
    return self.normal_dist.cdf(magnitude)


class GammaMagnitudeNormalizer(MagnitudeNormalizer):
  """Normalizer based on Gamma Distribution."""

  def __init__(self, gamma_shape: float, scale: float):
    """Initializer.

    Args:
      gamma_shape: Gamma shape param for Gamma.
      scale: scale param for Gamma.
    """
    super().__init__(scale=scale)
    self.gamma_dist = scipy.stats.gamma(gamma_shape)

  def __call__(self, magnitude: float) -> float:
    """Calculates probability of a given magnitude for a gamma distribution."""
    assert magnitude >= 0.0
    return self.gamma_dist.pdf(magnitude / self.scale)

  def cdf(self, magnitude: float) -> float:
    """Calculates the cumulative probability for a given magnitude."""
    assert magnitude >= 0.0
    return self.gamma_dist.cdf(magnitude / self.scale)


def pairwise(iterable):
  """From itertools: iterates pairwise over an iterable.

  Args:
    iterable: An iterable containing a sequence of items.

  Returns:
    An iterable containing pairs from the input iterable, e.g.
      s0,s1,s2,s3... -> (s0,s1), (s1,s2), (s2, s3), ...
  """
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


def get_scan_to_path_and_heading_dict(
    source_json_path: str) -> Dict[str, Sequence[data.Path]]:
  """Gets `scan` -> [data.Path(path, heading), ...] mapping.

  Helper function to collection transition stats house-wise.

  Args:
    source_json_path: Path to training R2R json.

  Returns:
    Mapping of scan -> [data.Path(path, heading), ...].
  """
  with tf.io.gfile.GFile(source_json_path, 'r') as fp:
    dataset = json.load(fp)
  scan_to_path_and_heading = collections.defaultdict(list)
  for item in dataset:
    path_container = data.Path(path=item['path'], heading=item['heading'])
    scan_to_path_and_heading[item['scan']].append(path_container)
  return scan_to_path_and_heading
