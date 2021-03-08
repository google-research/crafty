
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
"""Produces action sequences for panorama sequences in R2RItems."""

from typing import Optional, Sequence, Text

from crafty import data
from crafty import mp3d
from crafty import util

from valan.r2r import house_parser as rhp


class Walker:
  """Creates an action sequence given a path and associated scan data."""

  def __call__(
      self,
      scan_data: mp3d.ScanData,
      path: Sequence[Text],
      initial_heading: float,
      given_pano_objects: Optional[Sequence[rhp.RoomObject]] = None,
  ) -> Sequence[data.Motion]:
    """Get the heading changes and objects for the path in an R2R item.

    There is nothing fancy here; basically, this function just scoops up the
    information needed for later components.

    Args:
      scan_data: ScanData for the scan the path occurs in.
      path: the sequence of panoramas defining the path.
      initial_heading: the initial heading in the first panorama of the path.
      given_pano_objects: pre-selected pano objects.

    Returns:
      A sequence of Motion objects representing the contexts for moving along
      the given path.
    """
    # If pre-selected objects are provided, check 1:1 pano-object pairing.
    if given_pano_objects is not None:
      if len(given_pano_objects) != len(path):
        raise ValueError('#pano-objects must be equal to #panos.')
    curr_heading = initial_heading
    heading_sequence = [curr_heading]
    for i in range(len(path) - 1):
      pano_a, pano_b = path[i], path[i + 1]
      assert scan_data.is_valid_transition(pano_a, pano_b)

      # Update heading to continue on to next pano_to_pano connection.
      curr_heading = scan_data.graph.get_connections(pano_a)[pano_b].heading
      heading_sequence.append(curr_heading)
    # Repeat final heading as the 'exit' from last pano.
    heading_sequence.append(curr_heading)

    # Sliding window to get the entry/exit heading pairs for each pano.
    heading_changes = [
        data.HeadingChange(x, y) for x, y in util.pairwise(heading_sequence)
    ]

    pano_contexts = []
    for pano, heading_change in zip(path, heading_changes):
      center = scan_data.house.get_pano_by_name(pano).center
      # If pre-selected objects are provided, use them. Otherwise read in
      # all the available objects per pano for HMM to select.
      if given_pano_objects is None:
        pano_objs = scan_data.get_pano_object_list(pano)
      else:
        pano_objs = given_pano_objects
      pano_contexts.append(
          data.PanoContext(pano, center, heading_change, pano_objs))

    return get_motions(pano_contexts)


def get_motions(
    pano_contexts: Sequence[data.PanoContext]) -> Sequence[data.Motion]:
  """Constructs Motions from the PanoContexts.

  This is simple, basically just pairing PanoContexts as steps. However, a
  key aspect of this is that there is a Motion representing the beginning (for
  the initial heading into the first panorama and exiting it) and another Motion
  for the ending (with a dummy exit heading).

  Args:
    pano_contexts: The metadata collected for each panorama.

  Returns:
    The sequence of motions required to step from the start to the end.
  """
  motions = []
  for context_a, context_b in util.pairwise(pano_contexts):
    headchange_a = context_a.heading_change
    a_enter = headchange_a.init
    a_exit = headchange_a.end

    motions.append(data.Motion(context_a, context_a, a_enter))
    motions.append(data.Motion(context_a, context_b, a_exit))

  last_context = pano_contexts[-1]
  last_enter = last_context.heading_change.init

  motions.append(data.Motion(last_context, last_context, last_enter))
  return motions
