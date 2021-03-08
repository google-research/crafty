
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
"""Functions for working with Matterport 3D house data."""
import collections
import math
import os
from typing import Dict, Sequence, Set, Text

from crafty import data
import tensorflow as tf
from valan.datasets.common import graph_utils
from valan.r2r import house_parser as rhp
from valan.r2r import house_utils as r2r_utils


DirType = data.DirectionType

MATTERPORT_BASE_DIR = '/path/to/matterport_data'
# Src: https://github.com/niessner/Matterport/blob/master/data_organization.md
REGION_LABEL_TO_TEXT = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining room',
    'e': 'lobby',
    'f': 'family room',
    'g': 'garage',
    'h': 'hallway',
    'i': 'library',
    'j': 'laundryroom',
    'k': 'kitchen',
    'l': 'living room',
    'm': 'meeting room',
    'n': 'lounge',
    'o': 'office',
    'p': 'porch',
    'r': 'game room',
    's': 'stairs',
    't': 'toilet',
    'u': 'utility room',
    'v': 'tv',
    'w': 'gym',
    'x': 'outdoor',
    'y': 'balcony',
    'z': 'other room',
    'B': 'bar',
    'C': 'classroom',
    'D': 'dining booth',
    'S': 'spa',
    'Z': 'junk',
    '-': 'no label',
}


class MatterportData:
  """A class to support access various MP3D data elements."""

  def __init__(self, base_dir, occlusion_threshold=-1):
    self.base_dir: Text = base_dir
    self.occlusion_threshold = occlusion_threshold
    self.scans_dir: Text = os.path.join(base_dir, 'data', 'v1', 'scans')
    self.connections_dir = os.path.join(base_dir, 'data', 'v1', 'connections')

  def list_scans(self):
    """List the scans available in from the base directory."""
    return tf.io.gfile.listdir(self.scans_dir)

  def house_file_path(self, scan):
    """Gets the house file for the given scan."""
    return os.path.join(self.scans_dir, scan, 'house_segmentations',
                        f'{scan}.house')

  def connections_file_path(self, scan):
    """Gets the connectivity file for the given scan."""
    return os.path.join(self.connections_dir, f'{scan}_connectivity.json')

  def get_per_pano_objects(self, scans: Set[str]) -> Sequence[Sequence[Text]]:
    """For each pano in provided scans, extract clean category object names."""
    per_pano_objects = []
    for scan in scans:
      scan_data = ScanData(scan, self.base_dir, self.house_file_path(scan),
                           self.connections_file_path(scan),
                           self.occlusion_threshold)
      for pano in scan_data.graph.nodes:
        objects = scan_data.get_pano_object_list(pano)
        per_pano_objects.append([obj.clean_category for obj in objects])
    return per_pano_objects

  def get_scan_data(self, scan):
    return ScanData(scan, self.base_dir, self.house_file_path(scan),
                    self.connections_file_path(scan), self.occlusion_threshold)

  def get_pano_objects_in_scans(self, scans):
    """For each pano in provided scans, extract clean category object names."""
    all_pano_objects = []
    for scan in scans:
      scan_data = ScanData(scan, self.base_dir, self.house_file_path(scan),
                           self.connections_file_path(scan))
      all_pano_objects.append(scan_data.get_scan_objects())
    return all_pano_objects


class ScanData:
  """Holds house and graph representations and supports functions over them."""

  def __init__(self,
               scan_id: Text,
               base_dir: Text,
               house_file: Text,
               connections_file: Text,
               occlusion_threshold: float = -1):

    self.scan_id = scan_id
    self.house = rhp.R2RHouseParser(house_file, base_dir)
    self.graph = self.house.get_panos_graph(connections_file)
    self.occlusion_threshold = occlusion_threshold

    self.object_pano_distances = self.get_object_pano_distances()
    self.objects = sorted(self.object_pano_distances.keys())
    self.object_index = {o: i for i, o in enumerate(self.objects)}

    self.panos = sorted(list(self.graph.nodes))
    self.pano_index = {p: i for i, p in enumerate(self.panos)}

  def get_pano_object_list(self, pano: Text) -> Sequence[rhp.RoomObject]:
    """Gets all objects visible from the supplied panorama."""
    return list(
        self.house.get_pano_objects(pano, self.occlusion_threshold).values())

  def get_direction(self, pano_a, pano_b, heading=0):
    connections = self.graph.get_connections(pano_a)
    assert pano_b in connections
    return get_direction_type(connections[pano_b], heading)

  def is_valid_transition(self, pano_a, pano_b):
    return (pano_a in self.graph.nodes and pano_b in self.graph.nodes and
            pano_b in self.graph.nodes[pano_a].connections)

  def get_object_pano_distances(self) -> Dict[data.ObjectKey, Dict[str, float]]:
    """Given a scan, extract the distance of objects to each pano."""
    object_pano_distances = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for pano in self.graph.nodes:
      for obj in self.get_pano_object_list(pano):
        object_pano_distances[data.get_object_key(obj)][pano] = obj.distance
    return object_pano_distances

  def get_scan_objects(self):
    objects = set([])
    for pano in self.graph.nodes:
      pano_objects = self.house.get_pano_objects(pano).values()
      objects.update(set([o.clean_category for o in pano_objects]))
    return objects


def get_direction_type(connection_info, agent_heading):
  """Classifies a pano-to-pano graph ConnectionInfo into a DirectionType enum.

  Args:
    connection_info: `ConnectionInfo` object that indicates distance, heading
      and pitch between two connected graph nodes.
    agent_heading: The agent heading.

  Returns:
    One of enum `data.DirectionType`.
  """
  assert connection_info.heading >= 0. and connection_info.heading <= 2 * math.pi
  assert connection_info.pitch >= -math.pi and connection_info.pitch <= math.pi

  # TODO(eugeneie): update - between 'B' and 'A' is \pi (i.e.
  # `DirType.AROUND`) - between 'C' and 'B' is 3*\pi/2 (i.e.
  # `DirType.LEFT`), then the actual angle after an agent travels from
  # 'A' to 'B' is corrected - between 'C' and 'B' with agent's heading
  # coming from 'A' is \pi/2 (i.e. `DirType.RIGHT`).
  if connection_info.pitch >= math.pi / 12.:
    return DirType.UP
  elif connection_info.pitch <= -math.pi / 12.:
    return DirType.DOWN

  return get_heading_change_type(agent_heading, connection_info.heading)


def get_heading_change_type(goal_heading, base_heading):
  """Given a base heading and a goal heading, get the direction type."""
  # Ensure that the angles are always in [0, 2*pi].
  heading_diff = (2.0 * math.pi + goal_heading - base_heading) % (2.0 * math.pi)
  if (heading_diff <= math.pi / 8. or heading_diff >= 15. * math.pi / 8.):
    return DirType.STRAIGHT
  elif (heading_diff <= 3. * math.pi / 8. and heading_diff >= math.pi / 8.):
    return DirType.SLIGHT_RIGHT
  elif (heading_diff <= 6. * math.pi / 8. and
        heading_diff >= 3. * math.pi / 8.):
    return DirType.RIGHT
  elif (heading_diff <= 15. * math.pi / 8. and
        heading_diff >= 13. * math.pi / 8.):
    return DirType.SLIGHT_LEFT
  elif (heading_diff <= 13. * math.pi / 8. and
        heading_diff >= 10. * math.pi / 8.):
    return DirType.LEFT
  else:
    return DirType.AROUND


def get_connection_info(center1, center2):
  return graph_utils.ConnectionInfo(
      distance=r2r_utils.compute_distance(center1, center2),
      heading=r2r_utils.compute_heading_angle(center1, center2),
      pitch=r2r_utils.compute_pitch_angle(center1, center2))
