# FRANKA - TRAJECTORY GENERATION

Whistle is a software for music generation, allows you to train a neural network on any MIDI dataset.

![Trajectory](/images/trajectory.gif)

## Run the Software

### Requirements

* Python 3.x
* Installing the following packages using pip:
	* Keras
	* Tensorflow
	* h5py
    * Yaml
    * Numpy

### Set-Up the environment

- Clone this repo to your local machine using `git clone ...`.
- Insert your MIDI files into the folder `/midi_songs`.
- Edit the `config.yml` file with your preferences (you can leave the default settings).
- You can edit the network structure inside the file `whistle_train.py`.

### Train the Network

To train the network you have to run **train.py**.
```
python train.py
```
this will ...

### Run the Simulator

To run the network you have to run **whistle_run.py**. For the generation, the model **model.hdf5** will be used, so it will be necessary to rename in this way what one wants to use among the training ouput.
```
python whistle_run.py
```
After this we will have the **ouput.mid** file in the same foleder.

### Advice

...

## Built With

* [Python](https://www.python.org/)
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [Music21](http://web.mit.edu/music21/)

## Authors

* **Davide Corsi** - davide.corsi@univr.it

## License

- **MIT license**
- Copyright 2019 © **Davide Corsi** and **Enrico Marchesini**.

## Todos
- **TODO:** train to predict the note offset.
- **TODO:** add intro and conclusion for the song.
- **TODO:** implement a method for a non fixed offset between notes.
- **TODO:** testing other netork hyperparameters.
- **TODO:** try specific pattern generation as starting point for the network, instead a random sequence.

## Acknowledgments

We have to thank ...