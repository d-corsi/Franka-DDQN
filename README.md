# FRANKA - TRAJECTORY GENERATION

In this project I created a neural network for the generation of trajectories for the Franka manipulator. The generated model can then be used on the real robot via a ROS node, written in python with RosPy.

![Trajectory](/images/trajectory.gif)

## Run the Software

### Requirements

* Python 3.x
* Installing the following packages:
	* Keras, Tensorflow, h5py
    * Yaml, Numpy
	* ROS, Libfranka, RosPy

### Set-Up the environment

- Clone this repo to your local machine using `git clone ...`.
- Edit the `config.yml` file with your preferences (you can leave the default settings).
- You can edit the network structure inside the file `training.py`.

### Train the Network

To train the network you have to run **train.py**.
```
python training.py
```
this will generate a series of models whose success rate exceeds at least 90% of correct trajectories, these will will be saved in the **models** folder. The results of the training will be saved in the results folder, the most representative file is **reward_list.txt**. An example of plotted results is:

![Trajectory](/images/results_plot.png)

### Run the ROS node

- Install ROS and the Franka's libraries, both available on the official website.

The ROS node is contained in the **ros_node** folder and is written with RosPy. To run the software on the simulator, after launching RViz with the libfranka libraries, just run the command, where **model.h5** is the trained NN model:
```
roslaunch keras_pub.py model.h5
```
The ROS node is completely compatible with both the official visualizer and the real robot. In the figure below it is possible to see the comparison between the simulator and the robot.

![Trajectory](/images/simulator.gif)

## Built With

* [Python](https://www.python.org/)
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [ROS](http://www.ros.org/)

## Authors

* **Davide Corsi** - davide.corsi@univr.it
* **Enrico Marchesini** - enrico.marchesini@univr.it

## License

- **MIT license**
- Copyright 2019 © **Davide Corsi**.

## Acknowledgments

I have to thank the IT department of UNIVR and the Altair robotics lab for making the robot available to me.
