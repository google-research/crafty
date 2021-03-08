
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
"""Templatic instruction generation (English) for Matterport3D paths.

For an overview of the choices and flow in these templates, see the description
of the talker in https://arxiv.org/abs/2101.10504

Currently, the templates are just hacked together via Python functions and
string replacements. It would be nice to refactor this using a grammar and make
it all cleaner.

Eventually, Talker should be an interface, and the current implementation can
be something like TemplateTalker.
"""

import abc
import itertools
import random
from typing import Sequence, Set

import attr
from crafty import data
import numpy as np

DirType = data.DirectionType


@attr.s
class Talker(metaclass=abc.ABCMeta):
  """Base class for instruction generator.

  `merge_same_object_steps`: if true, the Talker merges subsequent
    ActionObservations that fixate on the same object.
  """

  merge_same_object_steps = attr.ib(default=True)

  def _get_observation_groups(
      self,
      action_observations: Sequence[data.ActionObservation],
  ) -> Sequence[Sequence[data.ActionObservation]]:
    if self.merge_same_object_steps:
      observation_groups = group_observation_steps(action_observations)
    else:
      observation_groups = [[ao] for ao in action_observations]
    return observation_groups

  @abc.abstractmethod
  def __call__(self):
    return

  @abc.abstractmethod
  def create_single_action_instruction(self):
    """Creates instruction for a single action."""
    return

  @abc.abstractmethod
  def create_multi_action_instruction(self):
    """Produces an instruction for multiple steps oriented on a common object."""
    return

  @abc.abstractmethod
  def create_instruction_step(self):
    """Helper function to invoke the right templates."""
    return


class TemplateTalker(Talker):
  """Templatic instruction generator based on actions and objects."""

  def __init__(self, merge_same_object_steps, mask_type=None):
    super().__init__(merge_same_object_steps)
    self.mask_type = mask_type

  def __call__(
      self,
      action_observations: Sequence[data.ActionObservation],
  ) -> Sequence[str]:
    """"Creates multistep instructions."""

    if self.mask_type == 'object':
      self._mask_objects_in_observations(action_observations)
    elif self.mask_type == 'direction':
      self._mask_directions_in_observations(action_observations)
    observation_groups = self._get_observation_groups(action_observations)

    all_instructions = []
    for group in observation_groups[:-1]:
      all_instructions.append(self.create_instruction_step(group))
    all_instructions.append(
        self.create_last_instruction(observation_groups[-1]))
    return all_instructions

  def _mask_objects_in_observations(
      self,
      action_observations: Sequence[data.ActionObservation],
  ):
    for i in range(len(action_observations)):
      old_object_key = action_observations[i].observation.object_key
      new_object_key = data.ObjectKey('OBJECT', old_object_key.location)
      action_observations[i].observation.object_key = new_object_key

  def _mask_directions_in_observations(
      self,
      action_observations: Sequence[data.ActionObservation],
  ):
    for i in range(len(action_observations)):
      action_observations[i].move_direction = DirType.NONE

  def create_single_action_instruction(
      self,
      ao: data.ActionObservation,
  ) -> str:
    """Creates instruction for a single action.

    Args:
      ao: A single ActionObservation to create an instruction for.

    Returns:
      A random choice from a set of possible instructions for the situation
      depicted in the given ActionObservation.
    """
    bp = BasePhrases.from_action_observation(ao)
    instructions = []
    if ao.move_direction == DirType.STOP:  # this is the first action
      instructions = [
          f'you should see a {bp.object} {bp.orientation}.',
          f'you are near a {bp.object}, {bp.orientation}.',
          f'{bp.orientation} there\'s a {bp.object}.',
          f'there is a {bp.object} {bp.orientation}.',
          f'there is a {bp.object} when you look {bp.orientation}.',
          f'a {bp.object} is {bp.orientation}.'
      ]
    else:
      if ao.move_type == 'intra':  # Movement within a panorama (turning).
        instructions = [
            f'{bp.movement}. you should see a {bp.object} {bp.orientation}.',
            f'{bp.movement}.',
            f'{bp.movement}. a {bp.object} is {bp.orientation}.',
            f'{bp.movement}. there is a {bp.object} {bp.orientation}.',
            f'you\'ll see a {bp.object} {bp.orientation} as you {bp.movement}.',
        ]
      else:  # 'inter' movement from one pano to another (stepping)
        instructions = [
            f'{bp.movement}, with the {bp.object} {bp.orientation}.',
            f'you\'ll see a {bp.object} {bp.orientation} as you {bp.movement}.',
            f'a {bp.object} is {bp.orientation} as you {bp.movement}.',
            f'{bp.movement}, going along to the {bp.object} {bp.orientation}.',
            f'{bp.movement}. a {bp.object} is {bp.orientation}.',
        ]
    return np.random.choice(instructions)

  def create_multi_action_instruction(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Produces an instruction for multiple steps oriented on a common object.

    Args:
      ao_group: a sequence of ActionObservations that (should) all center around
        a single object.

    Returns:
      A single instruction that covers all steps in the sequence.
    """
    if ao_group[0].move_direction == DirType.STOP:

      # This is the first sequence of ActionObservations in the journey.
      first_instruction = self.create_single_action_instruction(ao_group[0])
      bp = BasePhrases.from_action_observation(ao_group[1])
      second_instruction = f'{bp.movement}, so that it is {bp.orientation}.'
      return f'{first_instruction} {second_instruction}'

    else:

      # Handle direction changes separately so we can collapse multiple moves
      # of the same type. (And thus not say "walk forward and go ahead and ...")
      dir_changes = [
          x[0]
          for x in itertools.groupby([ao.move_direction for ao in ao_group])
      ]
      move_phrases = [sample_command(md) for md in dir_changes]
      move_command = ' and '.join(move_phrases)

      all_phrases = [BasePhrases.from_action_observation(ao) for ao in ao_group]

      final_obj_direction = ao_group[-1].obj_direction
      last_bp = all_phrases[-1]
      object_command = ''
      if final_obj_direction == DirType.STRAIGHT:
        object_command = f'heading toward the {last_bp.object}'
      elif final_obj_direction == DirType.AROUND:
        object_command = f'leaving the {last_bp.object} behind you'
      elif (final_obj_direction == DirType.SLIGHT_LEFT or
            final_obj_direction == DirType.SLIGHT_RIGHT):
        object_command = (f'approaching the {last_bp.object} '
                          f'{last_bp.orientation}')
      else:
        object_command = f'passing the {last_bp.object} {last_bp.orientation}'
      return move_command + ', ' + object_command + '.'

  def create_instruction_step(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Helper function to invoke the right templates."""
    if len(ao_group) == 1:
      return self.create_single_action_instruction(ao_group[0])
    else:
      return self.create_multi_action_instruction(ao_group)

  def create_last_instruction(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Creates the instruction for the last step.

    We treat this is a special case because we need to instruct the follower to
    end, rather than giving them orientation for continued navigation.

    Args:
      ao_group: a sequence of ActionObservations that (should) all center around
        a single object.

    Returns:
      A single instruction that covers all steps in the sequence.
    """

    bp = BasePhrases.from_action_observation(ao_group[-1])
    end_instructions = [
        f'{bp.movement} and stop by the {bp.object}.',
        f'{bp.movement}. stop when you get near the {bp.object}.',
        f'{bp.movement}. wait next to the {bp.object}.',
        f'{bp.movement}, then wait by the {bp.object}.',
        f'{bp.movement} to the {bp.object} and stop there.',
        f'{bp.movement} to the {bp.object}. you\'re done.',
    ]
    return np.random.choice(end_instructions)


class ObjectOnlyTalker(Talker):
  """Object-only instruction generator based only on selected landmarks."""

  def __call__(
      self,
      action_observations: Sequence[data.ActionObservation],
  ) -> Sequence[str]:

    observation_groups = self._get_observation_groups(action_observations)

    all_instructions = [
        self.create_instruction_step(group) for group in observation_groups
    ]
    return all_instructions

  def create_single_action_instruction(
      self,
      ao: data.ActionObservation,
  ) -> str:
    """Naive `instruction` step that only has a chosen object.

    Args:
      ao: A single ActionObservation to create an instruction for.

    Returns:
      A single-object/landmark `instruction`.
    """
    bp = BasePhrases.from_action_observation(ao)
    return f'{bp.object}.'

  def create_multi_action_instruction(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Similar to `single_action`, but collapses to get the last object.

    Args:
      ao_group: a sequence of ActionObservations that (should) all center around
        a single object.

    Returns:
      A single instruction that covers all steps in the sequence.
    """
    if ao_group[0].move_direction == DirType.STOP:
      return self.create_single_action_instruction(ao_group[0])
    else:
      return self.create_single_action_instruction(ao_group[-1])

  def create_instruction_step(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Helper function to invoke the right templates."""
    if len(ao_group) == 1:
      return self.create_single_action_instruction(ao_group[0])
    else:
      return self.create_multi_action_instruction(ao_group)


class DirectionOnlyTalker(Talker):
  """Direction-only instruction generator that does not use landmarks."""

  def __call__(
      self,
      action_observations: Sequence[data.ActionObservation],
  ) -> Sequence[str]:
    """"Creates multistep instructions."""

    observation_groups = self._get_observation_groups(action_observations)

    all_instructions = []
    for group in observation_groups[:-1]:
      all_instructions.append(self.create_instruction_step(group))
    all_instructions.append(
        self.create_last_instruction(observation_groups[-1]))
    return all_instructions

  def create_single_action_instruction(
      self,
      ao: data.ActionObservation,
  ) -> str:
    """Creates instruction for a single action.

    Args:
      ao: A single ActionObservation to create an instruction for.

    Returns:
      A random choice from a set of possible instructions for the situation
      depicted in the given ActionObservation.
    """
    bp = BasePhrases.from_action_observation(ao)
    if ao.move_direction == DirType.STOP:  # this is the first action
      # TemplateTalker typically says something like
      # `you should see OBJECT to your ORIENTATION` as a commence instruction.
      # In the direction-only condition, we simplified it all into a
      # placeholder token `BEGIN`.
      instruction = 'BEGIN'
    else:
      instruction = f'{bp.movement}.'
    return instruction

  def create_multi_action_instruction(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Produces an instruction for multiple steps oriented on a common object.

    Args:
      ao_group: a sequence of ActionObservations that (should) all center around
        a single object.

    Returns:
      A single instruction that covers all steps in the sequence.
    """
    if ao_group[0].move_direction == DirType.STOP:

      # This is the first sequence of ActionObservations in the journey.
      first_instruction = self.create_single_action_instruction(ao_group[0])
      bp = BasePhrases.from_action_observation(ao_group[1])
      second_instruction = f'{bp.movement}.'
      return f'{first_instruction} {second_instruction}'

    else:

      # Handle direction changes separately so we can collapse multiple moves
      # of the same type. (And thus not say "walk forward and go ahead and ...")
      dir_changes = [
          x[0]
          for x in itertools.groupby([ao.move_direction for ao in ao_group])
      ]
      move_phrases = [sample_command(md) for md in dir_changes]
      move_command = ' and '.join(move_phrases)

      return move_command

  def create_instruction_step(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Helper function to invoke the right templates."""
    if len(ao_group) == 1:
      return self.create_single_action_instruction(ao_group[0])
    else:
      return self.create_multi_action_instruction(ao_group)

  def create_last_instruction(
      self,
      ao_group: Sequence[data.ActionObservation],
  ) -> str:
    """Creates the instruction for the last step.

    We treat this is a special case because we need to instruct the follower to
    end, rather than giving them orientation for continued navigation.

    Args:
      ao_group: a sequence of ActionObservations that (should) all center around
        a single object.

    Returns:
      A single instruction that covers all steps in the sequence.
    """

    bp = BasePhrases.from_action_observation(ao_group[-1])
    return f'{bp.movement}.'


################################################################################
# Basic action commands.
################################################################################
SLIGHT_TURN_VERBS = frozenset(['bear', 'go slightly', 'curve'])
TURN_VERBS = frozenset(
    ['face', 'go', 'head', 'make a', 'pivot', 'turn', 'walk'])

SLIGHT_LEFT_COMMANDS = frozenset([f'{x} left' for x in SLIGHT_TURN_VERBS])
LEFT_COMMANDS = frozenset([f'{x} left' for x in TURN_VERBS])
SLIGHT_RIGHT_COMMANDS = frozenset([f'{x} right' for x in SLIGHT_TURN_VERBS])
RIGHT_COMMANDS = frozenset([f'{x} right' for x in TURN_VERBS])

MOVE_COMMANDS = frozenset(
    ['continue', 'go', 'head', 'proceed', 'walk', 'travel'])
STRAIGHT_MOVEMENT_WORDS = frozenset(['straight', 'forward'])

STRAIGHT_COMMANDS = frozenset([
    f'{x} {y}'
    for x, y in itertools.product(MOVE_COMMANDS, STRAIGHT_MOVEMENT_WORDS)
])

UP_MOVEMENT_WORDS = frozenset(
    ['up', 'upstairs', 'up the stairs', 'up the staircase'])
DOWN_MOVEMENT_WORDS = frozenset(
    ['down', 'downstairs', 'down the stairs', 'down the staircase'])
AROUND_COMMANDS = frozenset(['turn around'])

UP_COMMANDS = frozenset([
    f'{x} {y}' for x, y in itertools.product(MOVE_COMMANDS, UP_MOVEMENT_WORDS)
])
DOWN_COMMANDS = frozenset([
    f'{x} {y}' for x, y in itertools.product(MOVE_COMMANDS, DOWN_MOVEMENT_WORDS)
])

STOP_COMMANDS = frozenset(['stop there', 'stop', 'end', 'wait'])
################################################################################

################################################################################
# Basic orientation phrases.
################################################################################
DIRECTION_PRE = frozenset(['to your', 'to the', 'on your', 'on the'])
DIRECTION_POST = frozenset(['of you'])
SLIGHT_MODIFIERS = frozenset(['a bit', 'slightly', 'a little', 'just'])

AHEAD_PHRASES = frozenset(['ahead of you', 'in front of you'])
BEHIND_PHRASES = frozenset(['behind you', 'in back of you'])
UP_PHRASES = frozenset(['above you'])
DOWN_PHRASES = frozenset(['below you', 'at your feet'])
################################################################################


def sample_command(direction_type: DirType) -> str:
  """Randomly samples English expressions valid for the given direction type.

  Args:
    direction_type: The DirectionType to describe as a command.

  Returns:
    A random choice of available phrases for the given DirectionType.
  """
  if direction_type == DirType.NONE:
    return 'DIRECTION'  # generic mask for directions.
  elif direction_type == DirType.SLIGHT_LEFT:
    return random.sample(SLIGHT_LEFT_COMMANDS, 1)[0]
  elif direction_type == DirType.LEFT:
    return random.sample(LEFT_COMMANDS, 1)[0]
  elif direction_type == DirType.SLIGHT_RIGHT:
    return random.sample(SLIGHT_RIGHT_COMMANDS, 1)[0]
  elif direction_type == DirType.RIGHT:
    return random.sample(RIGHT_COMMANDS, 1)[0]
  elif direction_type == DirType.STRAIGHT:
    return random.sample(STRAIGHT_COMMANDS, 1)[0]
  elif direction_type == DirType.AROUND:
    return random.sample(AROUND_COMMANDS, 1)[0]
  elif direction_type == DirType.UP:
    return random.sample(UP_COMMANDS, 1)[0]
  elif direction_type == DirType.DOWN:
    return random.sample(DOWN_COMMANDS, 1)[0]
  elif direction_type == DirType.STOP:
    return random.sample(STOP_COMMANDS, 1)[0]
  else:
    raise ValueError('Unknown DirectionType:', direction_type)


def get_direction_phrases(direction: str, is_slight: bool = False) -> Set[str]:
  """Helper function to generate orientation commands.

  Args:
    direction: the string 'left' or 'right'
    is_slight: True if modifiers for being slightly to one side or the other
      should be applied, False if not.

  Returns:
    A sequence of possible direction phrases.
  """
  direction_phrases = set([f'{x} {direction}' for x in DIRECTION_PRE] +
                          [f'{direction} {x}' for x in DIRECTION_POST])
  if is_slight:
    return set([
        f'{x} {y}'
        for x, y in itertools.product(SLIGHT_MODIFIERS, direction_phrases)
    ])
  else:
    return direction_phrases


def sample_orientation(direction_type: DirType) -> str:
  """Randomly samples English expressions valid for the given direction type.

  Args:
      direction_type: The DirectionType to describe as a relative orientation.

  Returns:
    A random choice of available phrases for the given DirectionType.
  """
  if direction_type == DirType.SLIGHT_LEFT:
    return random.sample(get_direction_phrases('left', True), 1)[0]
  elif direction_type == DirType.LEFT:
    return random.sample(get_direction_phrases('left'), 1)[0]
  elif direction_type == DirType.SLIGHT_RIGHT:
    return random.sample(get_direction_phrases('right', True), 1)[0]
  elif direction_type == DirType.RIGHT:
    return random.sample(get_direction_phrases('right'), 1)[0]
  elif direction_type == DirType.STRAIGHT:
    return random.sample(AHEAD_PHRASES, 1)[0]
  elif direction_type == DirType.AROUND:
    return random.sample(BEHIND_PHRASES, 1)[0]
  elif direction_type == DirType.UP:
    return random.sample(UP_PHRASES, 1)[0]
  elif direction_type == DirType.DOWN:
    return random.sample(DOWN_PHRASES, 1)[0]
  elif direction_type == DirType.STOP:
    return 'here'
  else:
    raise ValueError('Unknown DirectionType:', direction_type)


@attr.s
class BasePhrases:
  """Phrase representations of components of an ActionObservation.

  `movement`: a description of the movement involved in the action
  `object`: a description of the object fixated on for this ActionObservation
  `orientation`: a phrase describing the orientation of the agent to the object
  """
  movement: str = attr.ib()
  object: str = attr.ib()
  orientation: str = attr.ib()

  @classmethod
  def from_action_observation(cls, ao: data.ActionObservation):
    return cls(
        movement=sample_command(ao.move_direction),
        object=ao.observation.object_key.category.replace('_', ' '),
        orientation=sample_orientation(ao.obj_direction))


def group_observation_steps(
    observations: Sequence[data.ActionObservation]
) -> Sequence[Sequence[data.ActionObservation]]:
  """Groups in-sequence actions oriented around a common object.

  Args:
    observations: the ActionObservations for an entire journey.

  Returns:
    The same observations, but grouped such that subsequent steps that involve
    the same object are together.
  """
  groups = []

  first_obs = observations[0]
  current_object = first_obs.observation.object_key
  current_group = [first_obs]

  for obs in observations[1:]:
    if current_object == obs.observation.object_key:
      current_group.append(obs)
    else:
      groups.append(current_group)
      current_group = [obs]
      current_object = obs.observation.object_key
  groups.append(current_group)

  return groups
