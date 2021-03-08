
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
"""A guide that can walk a path, observe objects and create VLN instructions.

The guide encompasses the walk, observe, and talk phases that collectively
implement a template-based instructor that has perfect perception.
"""

import attr

from crafty import mp3d
from crafty import observe
from crafty import util
from crafty import vln_data
from crafty import walk


@attr.s
class Guide:
  """A template-based instruction generator for vision-and-language navigation.

  The guide walks a path, creates an object observation sequence, and then
  produces an instruction based on positions, saliency and the required actions.

  `walker`: Collects the observations on the path.
  `talker`: Produces instructions based on the actions and observations.
  `hmm_type`: The HMM type. Currently Path-specific and Hard EM are available.
  """
  walker: walk.Walker = attr.ib()
  talker = attr.ib()
  observer = attr.ib()

  def __call__(
      self,
      item: vln_data.RRItem,
      scan_data: mp3d.ScanData,
  ) -> str:
    """Walk the path, select an object sequence, and then describe it all.

    This currently walks the path to collect available objects. It then
    randomly selects an object at each step, and uses a single template,
    "go to the OBJECT" for each step.

    Args:
      item: The R2RItem containing the panorama sequence to follow.
      scan_data: ScanData object holding relevant details and cached information
        about the scan (house) that this path traverses. This speeds up the
        observer considerably.

    Returns:
      A string with a full instruction for this path.
    """

    # Get the motions for traversing the path.
    if isinstance(item, vln_data.ExtendedRRItem):
      given_pano_objects = util.sample_given_pano_objects(item)
    else:
      given_pano_objects = None
    motions = self.walker(
        scan_data, item.path, item.heading, given_pano_objects)

    # Select an object for each motion using the observer.
    action_observations = self.observer(motions, scan_data)

    # Generate instructions using the talker.
    instructions = self.talker(action_observations)

    # Join the instructions to create a single instruction.
    return ' '.join(instructions)


def initialize_observer_with_hard_em(
    guide: Guide,
    train_data_path: str,
    mp_data: mp3d.MatterportData,
):
  """Train observer (with HardEMHMM) with the info from a training dataset.

  Args:
    guide: Guide runner.
    train_data_path: Path to a training JSON.
    mp_data: Matterport environment for loading scans.
  """
  assert isinstance(guide.observer, observe.HardEMObserver)
  scan_to_ph = util.get_scan_to_path_and_heading_dict(train_data_path)
  # Filter for the scans existing in the current train data.
  scans = [scan for scan in mp_data.list_scans() if scan in scan_to_ph]
  # TODO(wangsu) Adding facilities to allow sampling arbitrary paths.
  for scan in scans:
    scan_data = mp_data.get_scan_data(scan)
    motions_list = [
        guide.walker(scan_data, path_container.path, path_container.heading,
                     None)
        for path_container in scan_to_ph[scan]
    ]
    guide.observer.hard_em_trainer(motions_list, scan_data)
