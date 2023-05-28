import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10


SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('humanoid.xml'), common.ASSETS


@SUITE.add('benchmarking')
def collision(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Collision task."""
    physics = CollisionPhysics.from_xml_string(*get_model_and_assets())
    task = CollisionHumanoid(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class CollisionPhysics(mujoco.Physics):
    """Physics simulation with additional features for the Collision task."""

    def center_of_mass_positions(self):
        """Returns positions of the centers of mass for all entities."""
        return np.stack([self.named.data.subtree_com[entity] for entity in ['torso1', 'torso2']])

    def center_of_mass_velocities(self):
        """Returns velocities of the centers of mass for all entities."""
        return np.stack([self.named.data.subtree_comvel[entity] for entity in ['torso1', 'torso2']])


class CollisionHumanoid(base.Task):
    """A humanoid task with collision reward."""

    def __init__(self, move_speed, pure_state, random=None):
        """Initializes an instance of `CollisionHumanoid`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          pure_state: A bool. Whether the observations consist of the pure MuJoCo
            state or includes some useful features thereof.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        self._pure_state = pure_state
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        # Find a collision-free random initial configuration for each entity.
        penetrating = True
        while penetrating:
            for entity in ['torso1', 'torso2']:
                randomizers.randomize_limited_and_rotational_joints(physics, self.random, prefix=entity)
                # Check for collisions.
                physics.after_reset()
                penetrating = physics.data.ncon > 0
                if penetrating:
                    break
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns either the pure state or a set of egocentric features."""
        obs = collections.OrderedDict()
        if self._pure_state:
            obs['position'] = physics.position()
            obs['velocity'] = physics.velocity()
        else:
            obs['joint_angles'] = physics.joint_angles()
            obs['head_height'] = physics.head_height()
            obs['extremities'] = physics.extremities()
            obs['torso_vertical'] = physics.torso_vertical_orientation()
            obs['com_velocity'] = physics.center_of_mass_velocity()
            obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.head_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 4)
        upright = rewards.tolerance(physics.torso_upright(),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        small_control = rewards.tolerance(physics.control(), margin=1,
                                          value_at_margin=0,
                                          sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5

        collision_distance = np.linalg.norm(physics.center_of_mass_positions()[0] - physics.center_of_mass_positions()[1])
        collision_reward = rewards.tolerance(collision_distance,
                                             bounds=(0, 1),
                                             margin=0.1,
                                             sigmoid='linear',
                                             value_at_margin=1)

        if self._move_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move * collision_reward
        else:
            com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(com_velocity,
                                     bounds=(self._move_speed, float('inf')),
                                     margin=self._move_speed,
                                     value_at_margin=0,
                                     sigmoid='linear')
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move * collision_reward