
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
"""Observe the environment to produce object sequences given a path."""

import abc
import collections
import functools
import math
import random

from typing import Dict, Sequence, TypeVar

import attr

from crafty import data
from crafty import hmm
from crafty import mp3d
from crafty import util
from crafty.appraise import Appraiser
from valan.datasets.common import graph_utils
from valan.r2r import house_utils as r2r_utils

Tag = TypeVar('Tag')

_GAUSSIAN_MEAN_FOR_ALL_NORMALIZATION = 0.0
# Gives higher values for near distances that aren't in the agent's face.
_GAMMA_SHAPE_DISTANCE = 2.0
# For sweep and interest, we want higher values the closer the magnitude is to
# zero, which is accomplished by values less than one.
_GAMMA_SHAPE_SWEEP = 0.9
_GAMMA_SHAPE_INTEREST = 0.9


@attr.s
class Observer(metaclass=abc.ABCMeta):
  """An agent that follows paths and observes an object sequence.

  `mp_data`: MatterportData object enabling access to simulator information.
  `appraiser`: Appraiser object that gives a prominence score to each category
    of objects in the environment.
  `magnitude_normalization`: For calculating probability of magnitude.
  `distance_dev`: Parameter governing how far this Observer looks. Smaller
    values will give preference to objects near the Observer's position.
  `sweep_dev`: Parameter governing how likely the Observer is to notice objects
    given its current heading. Larger values will increase the likelihood of
    fixating on objects in its peripheral vision.
  `interest_dev`: Parameter governing how much the Observer cares about the
    Appraiser's scores. Larger values increase the impact of the Appraiser's
    scores on the Observer's preferences.
  `fixation_boost`: Parameter governing how much the Observer tends to remain
    fixated on a given object. The value itself is used as a boost to
    self-transitions in the HMM.
  """

  mp_data: mp3d.MatterportData = attr.ib()
  appraiser: Appraiser = attr.ib()
  magnitude_normalization: str = attr.ib()
  distance_dev: float = attr.ib()
  sweep_dev: float = attr.ib()
  interest_dev: float = attr.ib()
  fixation_boost: float = attr.ib()

  def __attrs_post_init__(self):
    """Selects a magnitude normalizer on initialization.

    Source: https://www.attrs.org/en/stable/init.html
    """
    # Note: *_loc are the location params for the chosen distribution.
    #       For Gaussian, it is the mean; for Gamma, gamma shape.
    self.magnitude_normalizer = functools.partial(
        util.MagnitudeNormalizer.create, self.magnitude_normalization)
    if self.magnitude_normalization == 'normal':
      self.dist_loc = _GAUSSIAN_MEAN_FOR_ALL_NORMALIZATION
      self.sweep_loc = _GAUSSIAN_MEAN_FOR_ALL_NORMALIZATION
      self.interest_loc = _GAUSSIAN_MEAN_FOR_ALL_NORMALIZATION
    elif self.magnitude_normalization == 'gamma':
      self.dist_loc = _GAMMA_SHAPE_DISTANCE
      self.sweep_loc = _GAMMA_SHAPE_SWEEP
      self.interest_loc = _GAMMA_SHAPE_INTEREST

    # Stores cached taggers for EM HMM.
    # Note: unlike PathSpecificHMM, where taggers are made on the fly,
    #       Hard EM based HMM are trained on per-house stats and cached
    #       for repeated use.
    self.scan_id_to_hmm = dict()

  @abc.abstractmethod
  def __call__(self):
    return

  def dist_prob(
      self,
      distance: float,
      modifier: float = 1.0,
  ) -> float:
    """Gives probability of a distance given the observer's distance deviation.

    Args:
      distance: the distance in meters to convert into a probability.
      modifier: a multiplicative parameter to modify the distance deviation of
        the observer for obtaining this probability. This allows one to scale
        the viewing distance in some cases without having another distance_dev
        parameter.

    Returns:
      A probability of the given distance obtained by normalizing its
        magnitude.
    """
    return self.magnitude_normalizer(self.dist_loc,
                                     self.distance_dev * modifier)(
                                         distance)

  def interest_prob(self, score: float) -> float:
    """Probability of an object being interesting given its score.

    Args:
      score: the interestingness score of the object.

    Returns:
      A probability of the given interestingness score obtained by normalizing
        its magnitude.
    """
    return 1 - self.magnitude_normalizer(self.sweep_loc, self.interest_dev)(
        score)

  def view_prob(
      self,
      view_diff: float,
      modifier: float = 1.0,
  ) -> float:
    """Gives probability of a given change in view, both up/down and sideways.

    Think of this as "how willing is the observer to swivel its head up, down,
    or around to observe objects within a given panorama.

    Args:
      view_diff: the radians difference in view between two directions, e.g.
        either side-to-side heading difference or up-to-down pitch difference.
        This mainly captures how much objects deviate from a level point-of-view
        in a given direction within a panorama.
      modifier: a multiplicative parameter to modify the sweep deviation of the
        observer for obtaining this probability. This allows one to scale the
        sweep change in some cases without having another sweep_dev parameter.

    Returns:
      A probability of the given distance obtained by normalizing its
        magnitude.
    """
    return self.magnitude_normalizer(self.interest_loc,
                                     self.sweep_dev * modifier)(
                                         view_diff)

  def get_object_pano_affinity(
      self,
      object_pano_distances: Dict[data.ObjectKey, Dict[str, float]],
  ) -> Dict[data.ObjectKey, Dict[str, float]]:
    """Computes distance-based affinity of objects to each pano.

    Args:
      object_pano_distances: dictionary containing the distances from each
        object to all the panoramas in the scan.

    Returns:
      A dictionary containing normalized distance scores for each object to all
      the panoramas in the scan. Note: these are not a probability distribution
      per object over panoramas, but instead each score for each object,
      panorama pair is [0,1].
    """
    object_pano_affinity = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for obj, pano_distances in object_pano_distances.items():
      for pano, distance in pano_distances.items():
        object_pano_affinity[obj][pano] = self.dist_prob(distance)
    return object_pano_affinity

  def get_prominence(
      self,
      pano_context: data.PanoContext,
      heading: float,
      path_category_counts: Dict[str, int],
  ) -> Dict[data.ObjectKey, float]:
    """Gets the visual prominence of each object in the given panorama.

    Args:
      pano_context: PanoContext object containing all the information for the
        panorama of interest.
      heading: The direction the agent is looking within the panorama.
      path_category_counts: The number of times a given category (e.g. 'couch')
        has been observed on this path.

    Returns:
      A dictionary giving a prominence probability distribution over all objects
      in the panorama.
    """
    pano_center = pano_context.center
    scores = []
    objects = []
    for obj in pano_context.objects:
      obj_heading = r2r_utils.compute_heading_angle(pano_center, obj.center)
      obj_pitch = r2r_utils.compute_pitch_angle(pano_center, obj.center)
      ppitch = self.view_prob(abs(obj_pitch), 0.5)
      pview = self.view_prob(abs(heading - obj_heading))
      pdist = self.dist_prob(obj.distance)
      path_prominence = path_prominence_factor(
          path_category_counts.get(obj.clean_category, 1))
      interestingness = self.appraiser(obj.clean_category,
                                       1.0 / path_prominence)
      pinterest = self.interest_prob(interestingness)
      scores.append(pview * pdist * pinterest * ppitch)
      objects.append(data.get_object_key(obj))

    total = sum(scores)
    if total:
      probs = [x / total for x in scores]
    else:
      probs = [1.0 / len(scores) for _ in scores]

    return dict(zip(objects, probs))

  def get_transitions_for_motion(
      self,
      motion: data.Motion,
      path_category_counts: Dict[str, int],
  ) -> Dict[data.ObjectKey, Dict[data.ObjectKey, float]]:
    """Gets object-object transition information between pano/heading pairs.

    Args:
      motion: Motion object representing the transition from one panorama to
        another.
      path_category_counts: The number of times a given category (e.g. 'couch')
        has been observed on this path.

    Returns:
      A transition pseudo-count matrix, represented as a dictionary from objects
        to a dictionary of scores for all other objects present in the panoramas
        represented in this motion.
    """
    prominence = self.get_prominence(motion.goal, motion.heading,
                                     path_category_counts)

    scores = collections.defaultdict(lambda: collections.defaultdict(float))
    for o1 in motion.source.objects:
      so1 = data.get_object_key(o1)
      for o2 in motion.goal.objects:
        so2 = data.get_object_key(o2)
        head_o1_o2 = r2r_utils.compute_heading_angle(o1.center, o2.center)
        head_o1_c2 = r2r_utils.compute_heading_angle(motion.source.center,
                                                     o2.center)
        dist_o1_to_c2 = r2r_utils.compute_distance(o1.center,
                                                   motion.goal.center)
        dist_o2_to_c2 = r2r_utils.compute_distance(o2.center,
                                                   motion.goal.center)

        pscan = (1 + math.cos((head_o1_o2 - motion.heading))) / 2
        pdist = self.dist_prob(dist_o1_to_c2 + dist_o2_to_c2, 2.0)
        pview = self.view_prob(abs(motion.heading - head_o1_c2), 2.0)

        scores[so1][so2] = pscan * pdist * pview * prominence[so2]

    return scores

  def get_step_transition_counts(
      self, motions: Sequence[data.Motion], path_category_counts: Dict[str, int]
  ) -> Sequence[Dict[Tag, Dict[Tag, float]]]:
    """Extracts transition information from motions (of a single path).

    Args:
      motions: A sequence of Motion object which represents the transition from
        one panorama to another.
      path_category_counts: The number of times a given category (e.g. 'couch')
        has been observed on this path.

    Returns:
      A sequence of transitions.
    """
    all_transitions = []
    for motion in motions:
      all_transitions.append(
          self.get_transitions_for_motion(motion, path_category_counts))
    return all_transitions

  def get_merged_transition_counts(
      self, motions_list: Sequence[Sequence[data.Motion]],
      path_category_counts_list: Sequence[Dict[str, int]]
  ) -> Sequence[Dict[Tag, Dict[Tag, float]]]:
    """Extracts transition information from motions (of a multiple paths).

    Args:
      motions_list: A list of `motions` sequences where each such sequence
        corresponds to a path.
      path_category_counts_list: A list of path_category_counts info, where each
        corresponds to a path.

    Returns:
      Merged transition information from all the paths in a single house.
    """
    merged_transition_dict = dict()
    for motions, path_category_counts in zip(motions_list,
                                             path_category_counts_list):
      step_transitions = self.get_step_transition_counts(
          motions, path_category_counts)
      for step_transition in step_transitions:
        merged_transition_dict.update(step_transition)
    return [merged_transition_dict]


class PathSpecificObserver(Observer):
  """An Observer agent that works off of path-specific HMM transitions.

  The class constructs an agent on the fly for each path based on the stats.
  """

  def __call__(
      self,
      motions: Sequence[data.Motion],
      scan_data: mp3d.ScanData,
  ) -> Sequence[data.ActionObservation]:
    """Traverses a path and produces an object observation sequence.

    Args:
      motions: The Motion sequence provided by a Walker.
      scan_data: Cached information about the scan, especially which objects are
        visible in each panorama.

    Returns:
      A sequence of ActionObservations that indicate actions (where each action
      is one or more pano-to-pano steps) and the objects the observer has
      fixated on during each action. For example, a single action could be two
      steps forward and one step to the left, fixating on a television, such
      that the multi-step instruction could be "head left before the TV" instead
      of the more verbose "go forward. go forward to the TV. turn left."
    """
    path_pano_objects = set()
    for motion in motions:
      for obj in motion.source.objects:
        path_pano_objects.add(data.get_object_key(obj))
    path_category_instances = [so.category for so in path_pano_objects]
    path_category_counts = collections.Counter(path_category_instances)

    raw_emission_counts = self.get_object_pano_affinity(
        scan_data.object_pano_distances)

    raw_transitions_counts = self.get_step_transition_counts(
        motions, path_category_counts)

    start_prominence = self.get_prominence(
        motions[0].source, motions[0].source.heading_change.init,
        path_category_counts)

    end_scores = {}
    final_objects = motions[-1].goal.objects
    for obj in final_objects:
      obj_key = data.get_object_key(obj)
      prob_of_distance = self.dist_prob(obj.distance)
      obj_category_count = path_category_counts.get(obj_key.category, 1)
      path_prominence = path_prominence_factor(obj_category_count)
      interestingness = self.appraiser(obj.category, 1.0 / path_prominence)
      prob_of_interest = self.interest_prob(interestingness)
      end_scores[obj_key] = prob_of_distance * prob_of_interest

    tagger = hmm.PathSpecificHMM(raw_emission_counts, start_prominence,
                                 raw_transitions_counts, end_scores,
                                 self.fixation_boost, 1e-10, 1e-10,
                                 scan_data.pano_index)

    path = [motion.source.pano for motion in motions]
    tag_sequence = tagger(path, path_pano_objects)

    observations = []
    for motion, tag in zip(motions, tag_sequence):
      observations.append(data.Observation(motion.source, motion.heading, tag))

    return build_action_observations(observations)


class HardEMObserver(Observer):
  """An Observer agent that works off of per-house stats.

  An agent here processes a path P with an HMM built from the path samples
  obtained for the house from which P is constructed. The per-house HMMs
  are only built once then used on all the individual paths from that house.
  """

  def hard_em_trainer(self, motions_list: Sequence[Sequence[data.Motion]],
                      scan_data: mp3d.ScanData) -> hmm.HardEMHMM:
    """Train an HMM with per-house stats and Hard EM.

    HardEM: https://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/em/em.pdf

    Args:
      motions_list: A list of Motion sequences provided by a Walker.
      scan_data: Cached information about the scan, especially which objects are
        visible in each panorama.

    Returns:
      Trained hmm.HardEMHMM tagger.
    """
    # If a tagger is computed for this scan data, return it.
    # Otherwise compute one and cache.
    scan_id = scan_data.scan_id
    if scan_id in self.scan_id_to_hmm:
      return self.scan_id_to_hmm[scan_id]

    path_category_counts_list = []
    path_pano_objects_list = []
    end_scores = {}
    for motions in motions_list:
      path_pano_objects = set()
      for motion in motions:
        for obj in motion.source.objects:
          path_pano_objects.add(data.get_object_key(obj))
      path_category_instances = [so.category for so in path_pano_objects]
      path_category_counts = collections.Counter(path_category_instances)
      path_category_counts_list.append(path_category_counts)
      path_pano_objects_list.append(path_pano_objects)

      final_objects = motions[-1].goal.objects
      for obj in final_objects:
        obj_key = data.get_object_key(obj)
        prob_of_distance = self.dist_prob(obj.distance)
        obj_category_count = path_category_counts.get(obj_key.category, 1)
        # TODO(wangsu) if EM across all paths were proven to work out well
        #              later, then drop this.
        path_prominence = path_prominence_factor(obj_category_count)
        interestingness = self.appraiser(obj.category, 1.0 / path_prominence)
        prob_of_interest = self.interest_prob(interestingness)
        end_scores[obj_key] = prob_of_distance * prob_of_interest

    raw_emission_counts = self.get_object_pano_affinity(
        scan_data.object_pano_distances)

    raw_transitions_counts = self.get_merged_transition_counts(
        motions_list, path_category_counts_list)

    start_prominence = self.get_prominence(
        motions_list[0][0].source,
        motions_list[0][0].source.heading_change.init, path_category_counts)

    # TODO(wangsu) address Jason's comments below in comparison experiment:
    # Open question: perhaps we can get away from these initializations
    # when we use EM? I was thinking we could seed all the emission
    # distributions as P(pano | object) being proportional to
    # distance(pano_i, object), and similarly P(pano_next | pano_prev)
    # is proportional to distance(pano_i, pano_prev). (Possibly using
    # the Gamma normalization in there.)
    tagger = hmm.HardEMHMM(raw_emission_counts, start_prominence,
                           raw_transitions_counts, end_scores,
                           self.fixation_boost, 1e-10, 1e-10,
                           scan_data.pano_index)

    # Initializes transitions and emissions with EM.
    paths = [
        [motion.source.pano for motion in motions] for motions in motions_list
    ]
    tagger.train(paths, path_pano_objects_list)

    self.scan_id_to_hmm[scan_id] = tagger

  def __call__(
      self,
      motions: Sequence[data.Motion],
      scan_data: mp3d.ScanData,
  ) -> Sequence[data.ActionObservation]:
    """Traverses a path and produces an object observation sequence.

    Args:
      motions: The Motion sequence provided by a Walker.
      scan_data: Cached information about the scan, especially which objects are
        visible in each panorama.

    Returns:
      A sequence of ActionObservations that indicate actions (same format as
      the output of `path_specific_call`).
    """
    assert scan_data.scan_id in self.scan_id_to_hmm
    tagger = self.scan_id_to_hmm[scan_data.scan_id]
    path = [motion.source.pano for motion in motions]
    path_pano_objects = set()
    for motion in motions:
      for obj in motion.source.objects:
        path_pano_objects.add(data.get_object_key(obj))
    tag_sequence = tagger(path, path_pano_objects)

    observations = []
    for motion, tag in zip(motions, tag_sequence):
      observations.append(data.Observation(motion.source, motion.heading, tag))

    return build_action_observations(observations)


class RandomSampleObserver(Observer):
  """An Observer agent that works off of randomly sampled landmark objects.

  The class randomly samples an object at each pano/path-point from the
  objects visible. To keep interface consistency with PathSpecific & HardEM
  Observers, the constructor here takes in the same args but do not act upon
  all of them.
  """

  def __call__(
      self,
      motions: Sequence[data.Motion],
      scan_data: mp3d.ScanData,
  ) -> Sequence[data.ActionObservation]:
    """Traverses a path and produces an object observation sequence.

    Args:
      motions: The Motion sequence provided by a Walker.
      scan_data: Cached information about the scan, especially which objects are
        visible in each panorama.

    Returns:
      A sequence of ActionObservations that indicate actions (where each action
      is one or more pano-to-pano steps) and the objects the observer has
      fixated on during each action. For example, a single action could be two
      steps forward and one step to the left, fixating on a television, such
      that the multi-step instruction could be "head left before the TV" instead
      of the more verbose "go forward. go forward to the TV. turn left."
    """
    tag_sequence = []
    for motion in motions:
      candidate_objects = []
      for obj in motion.source.objects:
        candidate_objects.append(data.get_object_key(obj))
      if candidate_objects:
        sample_object = random.choice(candidate_objects)
        tag_sequence.append(sample_object)

    observations = []
    for motion, tag in zip(motions, tag_sequence):
      observations.append(data.Observation(motion.source, motion.heading, tag))

    return build_action_observations(observations)


def path_prominence_factor(category_count):
  """Simply returns base-2 log of a category count."""
  return math.log(1 + category_count, 2)


def build_action_observations(
    observations: Sequence[data.Observation]
) -> Sequence[data.ActionObservation]:
  """Given observations, creates the actions associated with them.

  Args:
    observations: a sequence of Observation objects capturing a PanoContext, a
      heading, and the object the observer fixated on in that moment.

  Returns:
    A sequence of ActionObservations that contextualize each Observation in the
    panorama-to-panorama movements required to go from start to finish.
  """
  action_observations = []

  # Create the first ActionObservation.
  obs = observations[0]
  obj_connection = mp3d.get_connection_info(obs.pano_context.center,
                                            obs.object_key.location)
  obj_direction = mp3d.get_heading_change_type(obj_connection.heading,
                                               obs.heading)
  action_observations.append(
      data.ActionObservation(data.DirectionType.STOP, 'intra', obj_direction,
                             obs))

  # Create intermediate ActionObservations.
  prev_heading = obs.heading
  prev_pano_context = obs.pano_context
  prev_pano_center = prev_pano_context.center
  for obs in observations[1:]:
    pano_center = obs.pano_context.center
    panos_connection = graph_utils.ConnectionInfo(
        distance=r2r_utils.compute_distance(prev_pano_center, pano_center),
        heading=prev_heading,
        pitch=r2r_utils.compute_pitch_angle(prev_pano_center, pano_center))

    move_direction = mp3d.get_direction_type(panos_connection, obs.heading)

    obj_connection = mp3d.get_connection_info(pano_center,
                                              obs.object_key.location)
    obj_direction = mp3d.get_heading_change_type(obj_connection.heading,
                                                 obs.heading)
    move_type = 'intra' if obs.pano_context.pano == prev_pano_context.pano else 'inter'
    action_observations.append(
        data.ActionObservation(move_direction, move_type, obj_direction, obs))

    prev_heading = obs.heading
    prev_pano_context = obs.pano_context
    prev_pano_center = pano_center

  return action_observations
