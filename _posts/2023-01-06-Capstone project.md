---
layout: post
title: PyBullet-Gym for Drones 
author: [00953103 蘇冠豪、00953106 張恩碩、01053203 邱顯淳]
category: [Lecture]
tags: [jekyll, ai]
---

---
##  PyBullet-Gym for Drones (四軸無人機之強化學習)

**期末報告描述**

透過調整GUI按鍵數值(左右滑動)來控制四軸無人機飛行。

---

**開發過程:**
1. Cart-Pole System(車桿系統)<br>
2. 1D and 2D Quadrotor Systems(一維和二維四旋翼系統)<br>
3. Stabilization and Trajectory Tracking Tasks(穩定和軌跡跟踪任務)<br>
4. Safe-Control-Gym Extended API(安全控制運動擴展API)<br>
5. Computational Performance(計算性能)<br>

---

**執行環境:**
以下是使用版本供參考
* python 3.9.10
* gym-pybullet-drones-v0.5.2 (儲存.zip檔並解壓縮)
* PyCharm (選擇安裝程序中的所有選項並重新啟動)
* Visual Studio (選擇“使用 C++ 進行桌面開發”)

---

### PyCharm介紹

PyCharm: [PyCharm ](https://www.jetbrains.com/pycharm/)

**PyCharm** 是一個用於電腦編程的整合式開發環境，主要用於Python語言開發，由捷克公司JetBrains開發，提供代碼分析、圖形化除錯器，整合測試器、整合版本控制系統，並支援使用Django進行網頁開發。<br>
PyCharm是一個跨平台開發環境，擁有Microsoft Windows、macOS和Linux版本。社群版在Apache授權條款下釋出，另外還有專業版在專用授權條款下釋出，其擁有許多額外功能，比如Web開發、Python We框架、Python剖析器、遠端開發、支援資料庫與SQL等更多進階功能。

如果使用 Python 語言進行開發，PyCharm 支援下列幾種辨識功能
1. 項目和代碼導航：專門的項目視圖，文件結構視圖和文件、類、方法和用法之間的快速跳轉。<br>
2. Python 重構：包括重命名、提取方法、引入變量、引入常量、上拉、下壓等。<br>
3. 支持Web框架：Django，web2py和Flask。<br>
4. 集成的Python調試器。<br>
5. Google App Engine Python開發。<br>
6. 版本控制集成：Mercurial，Git，Subversion，Perforce和CVS的統一用戶界面，包含更改列表和合併。<br>
7. 它主要與許多其他面向Python的IDE競爭，包括Eclipse的PyDev和更廣泛的Komodo IDE。<br>

參考來源: [Python自習手札 ](https://ithelp.ithome.com.tw/articles/10196461)

---

### gym-pybullet-drones介紹

gym-pybullet-drones: [gym-pybullet-drones ](https://github.com/utiasDSL/gym-pybullet-drones)<br>
video: [Learning to Fly](https://www.youtube.com/watch?v=VdTsVu1HuYk&ab_channel=LearningSystemsandRoboticsLab)<br>
paper: [Safe Learning in Robotics](https://www.annualreviews.org/doi/abs/10.1146/annurev-control-042920-020211)<br>
paper: [Safe-Control-Gym](https://ieeexplore.ieee.org/abstract/document/9849119)<br>

**PyBullet** 是一個快速且易於使用的 Python 模塊，用於機器人仿真和機器學習，重點是模擬到真實的轉移。 使用 PyBullet，您可以從 URDF 加載關節體，SDF、MJCF 等文件格式。PyBullet 提供正向動力學模擬、逆向動力學計算、正向和逆向運動學、碰撞檢測和射線相交查詢。 Bullet Physics SDK 包括 PyBullet 機器人示例，例如模擬 Minitaur 四足動物、使用 TensorFlow 推理運行的人形機器人和 KUKA 手臂抓取物體。

**gym-pybullet-drones** 最近很多針對連續動作的 RL 研究都集中在策略梯度算法和 actor-critic 架構上。四旋翼飛行器是 (i) 一種易於理解的移動機器人平台，其 (ii) 控制可以構建為連續狀態和動作問題，但超出一維，(iii) 它增加了許多候選策略導致的複雜性不可恢復的狀態，違反了在蘊含的馬爾可夫鏈上存在靜止狀態分佈的假設。

|                                   | `gym-pybullet-drones` | [AirSim](https://github.com/microsoft/AirSim) | [Flightmare](https://github.com/uzh-rpg/flightmare) |
|---------------------------------: | :-------------------: | :-------------------------------------------: | :-------------------------------------------------: |
|                         *Physics* | PyBullet              | FastPhysicsEngine/PhysX                       | *Ad hoc*/Gazebo                                     |
|                       *Rendering* | PyBullet              | Unreal Engine 4                               | Unity                                               |
|                        *Language* | Python                | C++/C#                                        | C++/Python                                          |  
|           *RGB/Depth/Segm. views* | **Yes**               | **Yes**                                       | **Yes**                                             |
|             *Multi-agent control* | **Yes**               | **Yes**                                       | **Yes**                                             |
|                   *ROS interface* | ROS2/Python           | ROS/C++                                       | ROS/C++                                             |
|            *Hardware-In-The-Loop* | No                    | **Yes**                                       | No                                                  |
|         *Fully steppable physics* | **Yes**               | No                                            | **Yes**                                             |
|             *Aerodynamic effects* | Drag, downwash, ground| Drag                                          | Drag                                                |
|          *OpenAI [`Gym`](https://github.com/openai/gym/blob/master/gym/core.py) interface* | **Yes** | **[Yes](https://github.com/microsoft/AirSim/pull/3215)** | **Yes**                                             |
| *RLlib [`MultiAgentEnv`](https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py) interface* | **Yes** | No | No                           |

---

## 程式說明

程式的部分主要透過 [`gym`](https://gym.openai.com/docs/),  [`pybullet`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#), 
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html), and [`rllib`](https://docs.ray.io/en/master/rllib.html)去控制。<br>
影片錄製需要用到 [`ffmpeg`](https://ffmpeg.org)

---

### 程式碼

aer1216_fall2020_hw1_sim.py

```
import time
import random
import numpy as np
import pybullet as p

#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
"""import sys"""
"""sys.path.append('../')"""

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from aer1216_fall2020_hw1_ctrl import HW1Control

DURATION = 10
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    #### Create the ENVironment ################################
    ENV = CtrlAviary(gui=GUI, record=RECORD)
    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ)

    #### Initialize the controller #############################
    CTRL = HW1Control(ENV)

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()
    STATE = OBS["0"]["state"]
    ACTION["0"] = CTRL.compute_control(current_position=STATE[0:3],
                                       current_velocity=STATE[10:13],
                                       target_position=STATE[0:3],
                                       target_velocity=np.zeros(3),
                                       target_acceleration=np.zeros(3)
                                       )

    #### Initialize target trajectory ##########################
    TARGET_POSITION = np.array([[0, 0, 1.0] for i in range(DURATION*ENV.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * ENV.SIM_FREQ, 3])

    #### Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / ENV.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / ENV.SIM_FREQ

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, DURATION*ENV.SIM_FREQ):

        ### Secret control performance booster #####################
        # if i/ENV.SIM_FREQ>3 and i%30==0 and i/ENV.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        OBS, _, _, _ = ENV.step(ACTION)

        #### Compute control #######################################
        STATE = OBS["0"]["state"]
        ACTION["0"] = CTRL.compute_control(current_position=STATE[0:3],
                                           current_velocity=STATE[10:13],
                                           target_position=TARGET_POSITION[i, :],
                                           target_velocity=TARGET_VELOCITY[i, :],
                                           target_acceleration=TARGET_ACCELERATION[i, :]
                                           )

        #### Log the simulation ####################################
        LOGGER.log(drone=0, timestamp=i/ENV.SIM_FREQ, state=STATE)

        #### Printout ##############################################
        if i%ENV.SIM_FREQ == 0:
            ENV.render()

        #### Sync the simulation ###################################
        if GUI:
            sync(i, START, ENV.TIMESTEP)

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    LOGGER.save()

    #### Plot the simulation results ###########################
    LOGGER.plot()
```
DSLPIDControl.py(經過調整GUI按鍵**Use GUI RPM**數值(左右滑動)來控制四軸無人機飛行)
```
import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL by SiQi Zhou and James Xu.

    """

    ################################################################################

    def __init__(self,
                 env: BaseAviary
                 ):
        """DSL PID control initialization.

        Parameters
        ----------
        env : BaseAviary
            The simulation environment to control.

        """
        super().__init__(env=env)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_ang_vel=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_ang_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired angular velocity.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          target_ang_vel
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler,
                               target_ang_vel
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_ang_vel : ndarray
            (3,1)-shaped array of floats containing the desired angular velocity.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        ang_vel_e = target_ang_vel - cur_ang_vel
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, ang_vel_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(thrust):
        """Utility function interfacing 1, 2, or 3D use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()

```
fly.py(操控PID控制器飛行一無人機)

```
"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=3,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=5,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], INIT_XYZS[0, 2]
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(env) for i in range(ARGS.num_drones)]
    # ctrl = [SimplePIDControl(env) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], H+j*H_STEP])
                                                                       )

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], H+j*H_STEP, np.zeros(9)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(ARGS.num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()

```
compare.py(重播並比較儲存在example_trace.pkl檔中的軌跡)
```
"""Script comparing a gym_pybullet_drones simulation to a trace file.

It is meant to compare/validate the behaviour of a pyhsics implementation.

Example
-------
In a terminal, run as:

    $ python compare.py

Notes
-----
The comparison is along a 2D trajectory in the X-Z plane, between x == +1 and -1.

"""
import os
import time
import argparse
import pickle
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Trace comparison script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--physics',        default="pyb",               type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',            default=False,               type=str2bool,      help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--record_video',   default=False,               type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--trace_file',     default="example_trace.pkl", type=str,           help='Pickle file with the trace to compare to (default: "example_trace.pkl")', metavar='')
    ARGS = parser.parse_args()

    #### Load a trace and control reference from a .pkl file ###
    with open(os.path.dirname(os.path.abspath(__file__))+"/../files/"+ARGS.trace_file, 'rb') as in_file:
        TRACE_TIMESTAMPS, TRACE_DATA, TRACE_CTRL_REFERENCE, _, _, _ = pickle.load(in_file)

    #### Compute trace's parameters ############################
    DURATION_SEC = int(TRACE_TIMESTAMPS[-1])
    SIMULATION_FREQ_HZ = int(len(TRACE_TIMESTAMPS)/TRACE_TIMESTAMPS[-1])

    #### Initialize the simulation #############################
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=np.array([0, 0, .1]).reshape(1, 3),
                     physics=ARGS.physics,
                     freq=SIMULATION_FREQ_HZ,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=False
                     )
    INITIAL_STATE = env.reset()
    action = {"0": np.zeros(4)}
    pos_err = 9999.

    #### TRACE_FILE starts at [0,0,0], sim starts at [0,0,INITIAL_STATE[2]]
    TRACE_CTRL_REFERENCE[:, 2] = INITIAL_STATE["0"]["state"][2]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=SIMULATION_FREQ_HZ,
                    num_drones=2,
                    duration_sec=DURATION_SEC
                    )

    #### Initialize the controller #############################
    ctrl = DSLPIDControl(env)

    #### Run the comparison ####################################
    START = time.time()
    for i in range(DURATION_SEC*env.SIM_FREQ):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute next action using the set points from the trace
        action["0"], pos_err, yaw_err = ctrl.computeControlFromState(control_timestep=env.TIMESTEP,
                                                                     state=obs["0"]["state"],
                                                                     target_pos=TRACE_CTRL_REFERENCE[i, 0:3],
                                                                     target_vel=TRACE_CTRL_REFERENCE[i, 3:6]
                                                                     )

        #### Re-arrange the trace for consistency with the logger
        trace_obs = np.hstack([TRACE_DATA[i, 0:3], np.zeros(4), TRACE_DATA[i, 6:9], TRACE_DATA[i, 3:6], TRACE_DATA[i, 9:12], TRACE_DATA[i, 12:16]])

        #### Log the trace #########################################
        logger.log(drone=0,
                   timestamp=TRACE_TIMESTAMPS[i],
                   state=trace_obs,
                   control=np.hstack([TRACE_CTRL_REFERENCE[i, :], np.zeros(6)])
                   )

        #### Log the simulation ####################################
        logger.log(drone=1,
                   timestamp=i/env.SIM_FREQ,
                   state=obs["0"]["state"],
                   control=np.hstack([TRACE_CTRL_REFERENCE[i, :], np.zeros(6)])
                   )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0: 
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui: 
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Plot the simulation results ###########################
    logger.plot(pwm=True)

```
downwash.py(利用兩台無人機去測試downtest模型)
```
"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python downwash.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=10,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[.5, 0, 1],[-.5, 0, .5]])
    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=2,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=True
                     )

    #### Initialize the trajectories ###########################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [0.5*np.cos(2*np.pi*(i/NUM_WP)), 0]
    wp_counters = np.array([0, int(NUM_WP/2)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=ARGS.simulation_freq_hz,
                    num_drones=2,
                    duration_sec=ARGS.duration_sec
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(env) for i in range(2)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(2)}
    START = time.time()
    for i in range(ARGS.duration_sec*env.SIM_FREQ):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(2):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j, 2]])
                                                                       )

            #### Go to the next way point and loop #####################
            for j in range(2):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(2):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state=obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j ,2], np.zeros(9)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Plot the simulation results ###########################
    logger.plot()

```

---
### 系統測試及成果展示
aer1216_fall2020_hw1_sim.py<br>
<iframe width="702" height="395" src="https://www.youtube.com/embed/MqDBqMSzaEE" title="2023/01/04 aer1216_fall2020_hw1_sim_control" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><br><br>
![](https://github.com/Chiuuuuu/AI/blob/gh-pages/images/Figure_aer1216_fall2020_hw1_sim.png?raw=true)<br><br>

DSLPIDControl.py(control)<br>
<iframe width="885" height="498" src="https://www.youtube.com/embed/MqDBqMSzaEE" title="2023/01/04 aer1216_fall2020_hw1_sim_control" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><br><br>
![](https://github.com/Chiuuuuu/AI/blob/gh-pages/images/Figure_aer1216_fall2020_hw1_sim_control.png?raw=true)<br><br>

fly.py<br>
<iframe width="702" height="395" src="https://www.youtube.com/embed/Xg3rgkXIYWg" title="2023/01/06 fly.py" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><br><br>
![](https://github.com/Chiuuuuu/AI/blob/gh-pages/images/Figure_fly.png?raw=true)<br><br>

downwash.py<br>
<iframe width="702" height="395" src="https://www.youtube.com/embed/2FJSdsvM2-s" title="2023/01/06 downwash.py" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><br><br>
![](https://github.com/Chiuuuuu/AI/blob/gh-pages/images/Figure_downwash.png?raw=true)<br><br>

---

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

