
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
"""Hidden Markov Model implementation and helper functions."""

import abc
import math
from typing import Any, Dict, Optional, Sequence, Set, TypeVar

import numpy as np

Tag = TypeVar('Tag')


class BaseHMM(metaclass=abc.ABCMeta):
  """Base class for HMM."""

  def __init__(
      self,
      raw_emission_counts: Dict[Tag, Dict[str, float]],
      raw_initial_transition_counts: Dict[Tag, float],
      raw_step_transitions_counts: Sequence[Dict[Tag, Dict[Tag, float]]],
      raw_final_transition_counts: Dict[Tag, float],
      self_transition_boost: float,
      esmooth_lambda=1e-10,
      tsmooth_lambda=1e-10,
      vocab_index: Optional[Dict[str, int]] = None,
  ):
    """Create an HMM from raw count information and smoothing parameters.

    The inputs are several data structures containing raw, unnormalized counts.
    These are used to construct the emission and transition distributions of
    the HMM.

    Args:
      raw_emission_counts: Maps each tag to a dictionary representing some
        measure of the importance of each vocabulary symbol to the tag.
      raw_initial_transition_counts: Maps each tag to a measure of how likely it
        is to be the first in the sequence.
      raw_step_transitions_counts: A list of transitions, one for each step. For
        a given step, each tag is mapped to a dictionary representing some
        measure of how likely all other tags are to follow it.
      raw_final_transition_counts: Maps each tag to a measure of how likely it
        is to be the last in the sequence.
      self_transition_boost: A smoothing value to add to self transitions and
        thus increase the likelihood that the HMM stays fixated on a particular
        item for multiple steps.
      esmooth_lambda: Add-lambda smoothing parameter for emissions.
      tsmooth_lambda: Add-lambda smoothing parameter for transitions.
      vocab_index: Explicitly provided vocabulary index. This can be inferred
        from the raw emission counts if those counts are complete. Unfortunately
        panorama 4b643a114c19427ba3f1f5b2580f6724 in scan 3898 has no objects.
    """
    # Adding vars needed later.
    self.tsmooth_lambda = tsmooth_lambda
    self.esmooth_lambda = esmooth_lambda
    # Set up tag and vocab indices.
    tag_set = (
        raw_initial_transition_counts.keys()
        | raw_final_transition_counts.keys())
    for one_step_transitions in raw_step_transitions_counts:
      for tag, transition_dist in one_step_transitions.items():
        tag_set.add(tag)
        tag_set = tag_set.union(transition_dist.keys())
    self.tags = sorted(tag_set)
    self.tag_index = {tag: i for i, tag in enumerate(self.tags)}

    if vocab_index is not None:
      self.vocab_index = vocab_index
    else:
      vocab_set = set()
      for emission_dist in raw_emission_counts.values():
        vocab_set = vocab_set.union(emission_dist.keys())
      vocab = sorted(vocab_set)
      self.vocab_index = {word: i for i, word in enumerate(vocab)}

    # Set up emission probability matrix.
    self.bj = scores_to_logprob_matrix(raw_emission_counts, self.tag_index,
                                       self.vocab_index, esmooth_lambda)

    # Set up initial transition distribution: log P(tag | START)
    self.aij_init = scores_to_logprob_vec(raw_initial_transition_counts,
                                          self.tag_index, tsmooth_lambda)

    self.num_tags = len(self.aij_init)

    # Set up final transition distribution. Technically, this should be the
    # value log P(END | tag), but here we compute log P(tag | END). This choice
    # should be revisited.
    self.aij_end = scores_to_logprob_vec(raw_final_transition_counts,
                                         self.tag_index, tsmooth_lambda)

    self.self_transition_boost = self_transition_boost
    # Set up the transition distributions for each step. Note that unlike a
    # normal HMM which has a single transition distribution matrix governing all
    # steps, here we have a distinct transition distribution matrix for every
    # step. This allows us to capture the likelihood that, say, the observer
    # will fixate on a couch in the living room given that it was previously
    # fixated on the refrigerator in the kitchen -- and that the quantity
    # P(couch|refrigerator) would be very different if the transition occurred
    # between any other two panoramas.
    self.aij_steps = []
    for move_transitions in raw_step_transitions_counts:
      oo_transitions = scores_to_logprob_matrix(move_transitions,
                                                self.tag_index, self.tag_index,
                                                tsmooth_lambda)
      if self_transition_boost > 0:
        oo_transitions = boost_self_transitions(oo_transitions,
                                                self.self_transition_boost)
      self.aij_steps.append(oo_transitions)

  def viterbi_decode(self, path: Sequence[str],
                     objects: Set[Tag]) -> Sequence[Tag]:
    """Compute the optimal tag sequence using the Viterbi algorithm.

    This is a no frills implementation of Viterbi, following the notation in
    Fig 8.5 Jurafsky and Martin's textbook. See:
    https://web.stanford.edu/~jurafsky/slp3/8.pdf

    Args:
      path: The sequence of panorama observations.
      objects: The set of objects observable on this path.

    Returns:
      The optimal sequence of object observations given the model parameters.
    """
    valid_tags = np.array([self.tag_index[t] for t in objects])

    # indexed observations
    o = [self.vocab_index[pano] for pano in path]
    num_obs = len(o)

    # Viterbi score matrix
    v = np.zeros((self.num_tags, num_obs))

    # Backpointer matrix
    bp = np.zeros((self.num_tags, num_obs), np.int64)
    for tid in valid_tags:
      v[tid][0] = self.aij_init[tid] + self.bj[tid][o[0]]

    # oid is observation id, the index in the token sequence
    for oid in range(1, num_obs):
      # In the EM case, aij_steps only has 1 item.
      # which is made into self.aij = aij_steps[0].
      # The intention is to unify the interface.
      aij = self.aij_steps[oid - 1] if len(self.aij_steps) > 1 else self.aij
      for tcurr in valid_tags:
        scores = [v[tprev][oid - 1] + aij[tprev][tcurr] for tprev in valid_tags]
        max_score = max(scores)
        v[tcurr][oid] = max_score + self.bj[tcurr][o[oid]]
        prev_best_tag = valid_tags[scores.index(max_score)]
        bp[tcurr][oid] = prev_best_tag

    end_scores = [
        v[tprev][num_obs - 1] + self.aij_end[tprev] for tprev in valid_tags
    ]

    max_end_idx = np.argmax(end_scores)
    curr_best_tag = valid_tags[max_end_idx]

    # Accumulate the Viterbi sequence
    vseq = [curr_best_tag]
    score_seq = [max_score]
    for oid in range(num_obs - 1, 0, -1):
      curr_best_tag = bp[curr_best_tag][oid]
      curr_best_score = v[curr_best_tag][oid]
      vseq.append(curr_best_tag)
      score_seq.append(curr_best_score)

    # Reverse to get the forward optimal tag sequence and get the tags from
    # the tag indices.
    best_tag_seq = [self.tags[tag_index] for tag_index in vseq[::-1]]
    best_score_seq = score_seq[::-1]
    return best_tag_seq, best_score_seq

  @abc.abstractmethod
  def __call__(self):
    return


class PathSpecificHMM(BaseHMM):
  """A basic HMM implementation with step-specific transitions.

  This is a strange HMM as it is only good for a single path. Instead of having
  a single transition distribution governing all possible sequences, it has
  a specific tag transition distribution for each observation pair. For the
  Matterport3D setting, this captures the intuition that observing a given brown
  couch in one panorama has a biased distribution toward seeing other objects
  (including the same brown couch) after moving to the next panorama.

  In principle, we could preprocess a given scan and enumerate the transitions
  between every object for every pair of adjacent panoramas. The model defined
  in this way would then be able to produce object observations for any path in
  that scan. However, the construction of each path specific HMM is relatively
  quick and fine for current purposes.
  """

  def __init__(  # pylint:disable=useless-super-delegation
      self,
      raw_emission_counts: Dict[Tag, Dict[str, float]],
      raw_initial_transition_counts: Dict[Tag, float],
      raw_step_transitions_counts: Sequence[Dict[Tag, Dict[Tag, float]]],
      raw_final_transition_counts: Dict[Tag, float],
      self_transition_boost: float,
      esmooth_lambda=1e-10,
      tsmooth_lambda=1e-10,
      vocab_index: Optional[Dict[str, int]] = None):
    super().__init__(raw_emission_counts, raw_initial_transition_counts,
                     raw_step_transitions_counts, raw_final_transition_counts,
                     self_transition_boost, esmooth_lambda, tsmooth_lambda,
                     vocab_index)

  def __call__(self, path: Sequence[str], objects: Set[Tag]) -> Sequence[Tag]:
    """Compute the optimal tag sequence using the Viterbi algorithm."""
    predicted_objects, _ = self.viterbi_decode(path, objects)
    return predicted_objects


class HardEMHMM(BaseHMM):
  """A version of Hard-EM initialized HMM with per-house/scan transitions.

  The EM initialization takes stats from a provided sample of paths in a house,
  and apply to the computation for individual paths with the same
  full-house info.
  """

  # The difference here is, in EM case, the
  # raw_step_transitions_counts only has 1 step, but otherwise the same.
  def __init__(  # pylint:disable=useless-super-delegation
      self,
      raw_emission_counts: Dict[Tag, Dict[str, float]],
      raw_initial_transition_counts: Dict[Tag, float],
      raw_merged_transitions_counts: Sequence[Dict[Tag, Dict[Tag, float]]],
      raw_final_transition_counts: Dict[Tag, float],
      self_transition_boost: float,
      esmooth_lambda=1e-10,
      tsmooth_lambda=1e-10,
      vocab_index: Optional[Dict[str, int]] = None):
    super().__init__(raw_emission_counts, raw_initial_transition_counts,
                     raw_merged_transitions_counts, raw_final_transition_counts,
                     self_transition_boost, esmooth_lambda, tsmooth_lambda,
                     vocab_index)
    assert len(self.aij_steps) == 1
    self.aij = self.aij_steps[0]

  def train(self, paths, objects_list, max_iters=5):
    """Initialize an HMM with hard-EM training."""
    # Compute "number of tags" and "vocab size".
    num_objects = len(self.tag_index)
    num_panos = len(self.vocab_index)

    prev_scores = [-np.infty] * len(objects_list)

    for step in range(max_iters):

      # E-step:
      # This computes the empirical transition and emission counts.
      # using the Viterbi-decoded predictions off the initial params (aij & bj),
      # i.e. taking the observed transition and emission counts then
      # computing accordingly.
      # NB: in Soft-EM, we compute the *expected* counts rather than the
      #     actual/empirical counts.
      predicted_objects_list = []
      predicted_scores_list = []
      for path, objects in zip(paths, objects_list):
        predicted_objects, predicted_scores = self.viterbi_decode(path, objects)
        predicted_objects_list.append(predicted_objects)
        predicted_scores_list.append(predicted_scores)
      aij_hat = np.zeros_like(self.aij)
      bj_hat = np.zeros_like(self.bj)
      for predicted_objects, path in zip(predicted_objects_list, paths):
        object_indices = [self.tag_index[obj] for obj in predicted_objects]
        pano_indices = [self.vocab_index[pano] for pano in path]
        assert len(path) == len(pano_indices)

        for i in range(len(pano_indices) - 1):
          # Update transition counts.
          aij_hat[object_indices[i], object_indices[i + 1]] += 1
          # Update emission counts.
          bj_hat[object_indices[i], pano_indices[i]] += 1

      # M-step:
      # Using the empirical transition and emission counts collected in the
      # E-step to compute new params, i.e. new transition log-likelihood aij
      # and emission log-likelihood bj, which will be used in the next round
      # of E-step.
      updated_aij = np.log((aij_hat + self.tsmooth_lambda) / np.maximum(
          self.tsmooth_lambda * num_objects + np.sum(aij_hat, axis=1)[:, None],
          1))
      updated_bj = np.log((bj_hat + self.esmooth_lambda) / np.maximum(
          self.esmooth_lambda * num_panos + np.sum(bj_hat, axis=1)[:, None], 1))
      delta_aij = np.abs(updated_aij - self.aij).mean()
      delta_bj = np.abs(updated_bj - self.bj).mean()
      print(f'Step {step}, T: {delta_aij}, E: {delta_bj}.', flush=True)
      # Validating and printing scores.
      scores = [sum(predicted_scores)
                for prediced_scores in predicted_scores_list]
      assert len(predicted_scores_list) == len(prev_scores)
      if step > 0:
        for i in range(len(scores)):
          assert scores[i] >= prev_scores[i]
          prev_scores[i] = scores[i]
      print(' -- scores:', scores)

      self.aij = updated_aij
      self.bj = updated_bj

  def __call__(self, path: Sequence[str], objects: Set[Tag]) -> Sequence[Tag]:
    """Compute the optimal tag sequence using the Viterbi algorithm."""
    predicted_objects, _ = self.viterbi_decode(path, objects)
    return predicted_objects


def boost_self_transitions(
    transitions: np.ndarray,
    boost_amount: float,
) -> np.ndarray:
  """Increase the pseudo-counts of self-transitions by the provided amount.

  Args:
    transitions: matrix of log probabilities of tag transitions.
    boost_amount: pseudo-count to boost the probability of self transitions.

  Returns:
    A matrix of log probabilities, renormalized after boosting.
  """
  nrow, ncol = transitions.shape
  assert nrow == ncol
  boosted_transitions = np.zeros((nrow, ncol))
  for x in range(nrow):
    probvec = [math.exp(transitions[x][y]) for y in range(ncol)]
    probvec[x] += boost_amount
    total = sum(probvec)
    boosted_transitions[x] = [math.log(y / total) for y in probvec]
  return boosted_transitions


def scores_to_logprob_matrix(
    per_x_scores: Dict[Any, Dict[Any, float]],
    row_index: Dict[Any, int],
    col_index: Dict[Any, int],
    add_lambda: float = 1.0,
) -> np.ndarray:
  """Create log probability matrix from scores of y's given x's."""
  logprob_matrix = np.zeros((len(row_index), len(col_index)))
  for x_val, x_id in row_index.items():
    logprob_matrix[x_id] = scores_to_logprob_vec(per_x_scores[x_val], col_index,
                                                 add_lambda)
  return logprob_matrix


def scores_to_logprob_vec(
    scores: Dict[Any, float],
    outcomes_index: Dict[Any, int],
    add_lambda: float = 1.0,
) -> np.ndarray:
  """Create probability vector from scores of a set of outcomes."""
  num_outcomes = len(outcomes_index)
  pvec = np.zeros(num_outcomes)
  denominator = sum(scores.values()) + add_lambda * num_outcomes
  for y_val, y_id in outcomes_index.items():
    y_score = scores.get(y_val, 0.0)
    pvec[y_id] = math.log((y_score + add_lambda) / denominator)
  return pvec
