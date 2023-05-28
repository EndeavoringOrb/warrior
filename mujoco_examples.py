import time
import os

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)

#output space 21 actuators
"""
body_pos
body_inertia
actuator_gear
actuator_acc0
actuator_ctrlrange

these are constant
print(m.body_pos)
print(m.body_inertia)
print(m.actuator_acc0)

model input:
 - qpos
 - qvel

model output:
 - ctrl

input into d.ctrl
output is d.actuator_force
"""

runtime = float('inf')

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < runtime:
    
    #print(d.ctrl,end="\r")
    #print(d.xpos)
    #print(f"head z: {d.xpos[2][2]}")
    #print(f"top torso z: {d.xpos[1][2]}")
    #upright_reward = (d.xpos[2][2]-d.xpos[1][2]) * (1/0.19)
    #print(upright_reward)
    os.system('cls')
    print(d.xpos)

    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)