
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
"""Container objects for passing data between Crafty components."""

import enum
from typing import Sequence, Text, Tuple

import attr

from valan.r2r import house_parser as rhp


@attr.s(frozen=True)
class ObjectKey:
  """A hashable representation of a house object.

  `category` is the object's category. This is the clean_category if built using
  the helper function get_object_key().

  `location` is the 3D point in the scan corresponding to the object's center.
  """
  category: Text = attr.ib()
  location: Tuple[float, float, float] = attr.ib()


def get_object_key(obj: rhp.RoomObject):
  """Create an ObjectKey from a RoomObject.

  Args:
    obj: A RoomObject in an R2R scan.

  Returns:
    An ObjectKey using the clean category and object center.
  """
  return ObjectKey(obj.clean_category, obj.center)


@attr.s
class HeadingChange:
  """A pair of headings representing how one enters and exits a panorama.

  `init`: the heading as one enters a panorama.
  `end`: the heading as one leaves a panorama.
  """
  init: float = attr.ib()
  end: float = attr.ib()


@attr.s
class PanoContext:
  """A collection of data relevant to passing through a panorama.

  `pano`: the panorama being passed through.
  `center`: the center of the panorama.
  `heading_change`: the entry and exit headings when passing through.
  `objects`: all the RoomObjects visible from the center of this panorama.
  """
  pano: str = attr.ib()
  center: Tuple[float, float, float] = attr.ib()
  heading_change: HeadingChange = attr.ib()
  objects: Sequence[rhp.RoomObject] = attr.ib()


@attr.s
class Motion:
  """The context as one moves from one panorama to another.

  `source`: the panorama one starts from for this step.
  `goal`: the panorama one ends in for this step.
  `heading`: the heading between the centers of `source` and `goal`
  """
  source: PanoContext = attr.ib()
  goal: PanoContext = attr.ib()
  heading: float = attr.ib()


class DirectionType(enum.IntEnum):
  """An enum indicating different directions between pano-to-pano movements."""
  UNSPECIFIED = 0
  SLIGHT_LEFT = 1
  LEFT = 2
  SLIGHT_RIGHT = 3
  RIGHT = 4
  STRAIGHT = 5
  AROUND = 6
  UP = 7
  DOWN = 8
  STOP = 9
  NONE = 10


@attr.s
class Observation:
  """An observation of a object in a particular panorama.

  `pano_context`: PanoContext information for the panorama this observation
    occurred in.
  `heading`: The direction of leaving this panoramo to go to the next.
  `object_key`: The ObjectKey of the object selected for this observation. This
    key corresponds to one of the RoomObjects in pano_context.objects.
  """
  pano_context: PanoContext = attr.ib()
  heading: float = attr.ib()
  object_key: ObjectKey = attr.ib()


@attr.s
class ActionObservation:
  """An Observation contextualized in its action context.

  `move_direction`: The DirectionType involved in the action.
  `move_type`: Indicates whether this is a move within a panorama ('intra') or
    across panoramas ('inter').
  `obj_direction` The heading of the object from the center of the panorama in
    the observation.
  `observation`: The panorama, heading and observed object at this position.
  """
  move_direction: DirectionType = attr.ib()
  move_type: Text = attr.ib()
  obj_direction: DirectionType = attr.ib()
  observation: Observation = attr.ib()


@attr.s
class Path:
  """An object encapsulating the path and the head in a data entry.

  `path`: A sequence of string identifier for the panoramas along a path.
  `heading`: Initial heading.
  """
  path: Sequence[str] = attr.ib()
  heading: float = attr.ib()


@attr.s
class RegionWithObjects:
  region_name: str = attr.ib()
  object_names: Sequence[str] = attr.ib()
