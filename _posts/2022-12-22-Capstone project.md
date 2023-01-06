---
layout: post
title: PyBullet-Gym for Drones 
author: [johnny, su, chiu]
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

程式的部分主要分成**辨識手掌**和**產生泡泡**還有**計分計時**三個部分，其中辨識手掌的部分主要參考自MediaPipe的官方文檔如下。<br>
程式的部分主要透過 [`gym`](https://gym.openai.com/docs/),  [`pybullet`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#), 
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html), and [`rllib`](https://docs.ray.io/en/master/rllib.html)去控制。<br>
影片錄製需要用到 [`ffmpeg`](https://ffmpeg.org)

---

### 官方文檔

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
aer1216_fall2020_hw2_sim.py

```
import time
import random
import numpy as np
import pybullet as p

#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
# import sys
# sys.path.append('../')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from aer1216_fall2020_hw2_ctrl import HW2Control

DURATION = 30
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    #### Create the ENVironment ################################
    ENV = CtrlAviary(num_drones=3,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, .0, .15], [-.3, .0, .15], [.3, .0, .15] ]),
                     gui=GUI,
                     record=RECORD
                     )
    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ,
                    num_drones=3,
                    )

    #### Initialize the CONTROLLERS ############################
    CTRL_0 = HW2Control(env=ENV,
                        control_type=0
                        )
    CTRL_1 = HW2Control(env=ENV,
                        control_type=1
                        )
    CTRL_2 = HW2Control(env=ENV,
                        control_type=2
                        )

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()
    STATE = OBS["0"]["state"]
    ACTION["0"] = CTRL_0.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         current_rpy_dot=STATE[13:16],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS["1"]["state"]
    ACTION["1"] = CTRL_1.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         current_rpy_dot=STATE[13:16],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS["2"]["state"]
    ACTION["2"] = CTRL_2.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         current_rpy_dot=STATE[13:16],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    #### Initialize the target trajectory ######################
    TARGET_POSITION = np.array([[0, 4.0*np.cos(0.006*i), 1.0] for i in range(DURATION*ENV.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * ENV.SIM_FREQ, 3])

    #### Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :])/ENV.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / ENV.SIM_FREQ

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, DURATION*ENV.SIM_FREQ):

        ### Secret control performance booster #####################
        # if i/ENV.SIM_FREQ>3 and i%30==0 and i/ENV.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        OBS, _, _, _ = ENV.step(ACTION)

        #### Compute control for drone 0 ###########################
        STATE = OBS["0"]["state"]
        ACTION["0"] = CTRL_0.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             current_rpy_dot=STATE[13:16],
                                             target_position=TARGET_POSITION[i, :],
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 0 ###########################################
        LOGGER.log(drone=0, timestamp=i/ENV.SIM_FREQ, state=STATE)

        #### Compute control for drone 1 ###########################
        STATE = OBS["1"]["state"]
        ACTION["1"] = CTRL_1.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             current_rpy_dot=STATE[13:16],
                                             target_position=TARGET_POSITION[i, :] + np.array([-.3, .0, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 1 ###########################################
        LOGGER.log(drone=1, timestamp=i/ENV.SIM_FREQ, state=STATE)

        #### Compute control for drone 2 ###########################
        STATE = OBS["2"]["state"]
        ACTION["2"] = CTRL_2.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             current_rpy_dot=STATE[13:16],
                                             target_position=TARGET_POSITION[i, :] + np.array([.3, .0, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 2 ###########################################
        LOGGER.log(drone=2, timestamp=i/ENV.SIM_FREQ, state=STATE)

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


## Student's Accomplishment
### [MediaPipe Hands poke bubbles](https://hahakevin45.github.io/AI/lecture/2022/12/08/Pose-Estimation.html)
<iframe width="854" height="480" src="https://www.youtube.com/embed/YJ_JCDBOgiE" title="MediaPipe Hands poke bubbles" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [使用 YOLOv5辨識水下魚種](https://alanlee0323.github.io/AI-course/lecture/2022/12/08/capstone-project.html)
<iframe width="723" height="482" src="https://www.youtube.com/embed/46wfrbQC8fI" title="影像偵測1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Teacher's samples

### Highway Traffice Analysis
**Kaggle:** [YOLOv5 Traffic Analysis](https://kaggle.com/rkuo2000/yolov5-traffic-analysis)<br>
use YOLOv5 to detect car/truck per frame, then analyze vehicle counts per lane and the estimated speed
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_traffic_analysis.jpg?raw=true)

---
### Facemask Detection
**Kaggle**: [YOLOv5 Facemask](https://www.kaggle.com/code/rkuo2000/yolov5-facemask)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Facemask.jpg?raw=true)

---
### Face Identification
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>

---
### DeepFashion TryOn
**Kaggle:** [rkuo2000/deepfashion-tryon](https://www.kaggle.com/rkuo2000/deepfashion-tryon)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/VTON_ACGPN.jpg?raw=true)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

---
### 系統測試及成果展示
<iframe width="664" height="498" src="https://www.youtube.com/embed/OP5HcXJg2Aw?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J" title="【機器學習2021】卷積神經網路 (Convolutional Neural Networks, CNN)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
<br>


**專題實作步驟:**
1. 建立身體動作之姿態照片資料集 (例如：5 poses , take 20 pictures of each pose)<br>
2. 始用**MMPose** 辨識出照片中的各姿勢之身體關鍵點 (use MMPose convert 16 keypoints (x,y) of each pose)<br>
3. 產生姿態關鍵點資料集 x_train.append(pose_keypoints) ( x_train.shape = (20x5, 16, 2), y_train.shape= (20x5, 1) )<br>
4. 建立DNN模型並訓練模型, 然後下載模型檔`pose_dnn.h5`至PC <br>
5. 於PC建立帶camera輸入之服務器程式, 載入模型`pose_dnn.h5`進行姿態動作辨識 <br>

**模型建構與訓練之程式樣本** (PC or Kaggle)<br>

```
input_shape=(16,2)
num_classes=5

inputs = layers.Input(shape=input_shape)
x = layers.Dense(128)(inputs)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=inputs, outputs=outputs)

models.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size=1, epochs=20, validation_data=(x_test, y_test))
models.save_model(model, 'pose_dnn.h5')
```

**姿態辨識服務器之程式樣本** (PC with Camera)<br>

```
model = models.load_model('models/pose_dnn.h5')
labels = ['stand', 'raise-right-arm', 'raise-left-arm', 'cross arms','both-arms-left']

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    mmdet_results = inference_detector(det_model, image) # 人物偵測產生BBox
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id) # 記住人物之BBox  
    pose_results, returned_outputs = inference_top_down_pose_model(...) # 感測姿態產生pose keypoints
    
    x_test = np.array(preson_results).reshape(1,16,2) # 將Keypoints List 轉成 numpy Array
    preds = model.fit(x_test) # 辨識姿態動作
    maxindex = int(np.argmax(preds))
    txt = labels[maxindex]
    print(txt)
```



---
### Hand Pose
![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)
**Dataset:** [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)<br>

1. Download pre-trained InterNet from [here](https://drive.google.com/drive/folders/1BET1f5p2-1OBOz6aNLuPBAVs_9NLz5Jo?usp=sharing)
2. Put the model at `demo` folder
3. Go to `demo` folder and edit `bbox` in [here](https://github.com/facebookresearch/InterHand2.6M/blob/5de679e614151ccfd140f0f20cc08a5f94d4b147/demo/demo.py#L74)
4. run `python demo.py --gpu 0 --test_epoch 20`
5. You can see `result_2D.jpg` and 3D viewer.

**Camera positios visualization demo**
1. `cd tool/camera_visualize`
2. Run `python camera_visualize.py`


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

