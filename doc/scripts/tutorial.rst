Tutorial
========

To predict spikes from calcium traces we are going to run commands like the following:

.. code-block:: bash

    $ c2s predict data.mat predictions.mat

The following tutorial describes how to format and preprocess your data, and how to improve and
evaluate predictions.

Data formatting
---------------

Inputs and outputs can be stored in either MATLAB or `Python <https://docs.python.org/2/library/pickle.html>`_ format.

Using MATLAB
~~~~~~~~~~~~

When using MATLAB files, data needs to be stored in a cell array named ``data``. Each entry of the cell array
should be a ``struct`` containing at least the fields ``calcium`` and ``fps``. Accessing the 12th
entry of the cell array might present you with something like the following output:

.. code-block:: matlab

    >> data{12}
    ans =
         calcium: [1x71985 double]
          spikes: [1x71985 uint16]
             fps: 99.9998
        cell_num: 8

Here, ``calcium`` is a 1xT vector containing the
calcium or fluorescence trace, while ``fps`` is a ``double`` referring to the corresponding sampling rate
in bins per second. Optionally, each entry may contain the fields ``spikes``, ``spike_times``, and
``cell_num``. The field ``spikes`` should correspond to a binned spike train at the same sampling
rate as the calcium trace. However, the preferred method is to specify the spikes times in
milliseconds via ``spike_times``, since spikes are typically recorded at much higher sampling rates.
In this case the output might look like:

.. code-block:: matlab

    >> data{12}
    ans =
            calcium: [1x71985 double]
        spike_times: [1x1202 double]
                fps: 99.9998
           cell_num: 8

The spikes can be used for training a statistical model to predict spikes. The ``cell_num`` field
is used to group recordings together and will affect the leave-one-out cross-validation.
This is useful, for example, if data from one cell was split into two or more sessions. If each entry
corresponds to a different cell, this field can be ignored.


Using Python
~~~~~~~~~~~~

In Python, data should be stored in lists of dictionaries and saved as `pickled objects <https://docs.python.org/2/library/pickle.html>`_.
Each dictionary element should contain at least the entries ``calcium`` and ``fps``. Accessing the 12th
entry of the list might present you with something like the following output:

>>> data[12]
{'calcium': array([[ 0.391,  0.490, ...,  0.221,  0.307]]),
 'fps': 99.9998,
 'cell_num': 8,
 'spikes': array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint16)}

Here, ``calcium`` is a 1xT NumPy array containing the calcium or fluorescence trace, while ``fps``
is a float value referring to the corresponding sampling rate in bins per second. Optionally,
each dictionary may contain the entries ``spikes``, ``spike_times``, and ``cell_num``. The entry
``spikes`` should correspond to a binned spike traini at the same sampling rate as the calcium
trace. However, the preferred method is to specify the spikes times in milliseconds via
``spike_times``, since spikes are typically recorded at much higher sampling rates.
In this case the output might look like:

>>> data[12]
{'calcium': array([[ 0.391,  0.490, ...,  0.221,  0.307]]),
 'fps': 99.9998,
 'cell_num': 8,
 'spike_times': array([[ 5951.34, 6007.95, ..., 719155.46, 719307.52]])}

The spikes can be used for training a statistical model to predict spikes. The ``cell_num`` entry
is used to group recordings together and will affect the leave-one-out cross-validation.
This is useful, for example, if data from one cell was split into two or more sessions. If each entry
corresponds to a different cell, this field can be ignored. To save the data, use ``pickle``,

>>> from pickle import dump
>>> with open('data.pck') as handle:
>>>    dump(data, handle, protocol=2)


Preprocessing
-------------

After the data has been brought into the right format, we should preprocess it.

.. code-block:: bash

    $ c2s preprocess data.pck data.preprocessed.pck

If your data is stored in MATLAB files, use

.. code-block:: bash

    $ c2s preprocess data.mat data.preprocessed.mat

The desired format is automatically inferred from the file ending. The preprocessing
tries to remove linear trends from the calcium trace and up- or downsamples the data so that all
traces have the same sampling rate. By default, this sampling rate is 100 fps but can be changed
with

.. code-block:: bash

    $ c2s preprocess --fps 100 data.mat data.preprocessed.mat

to something else if desired. Additionally, the preprocessing computes ``spikes`` from
``spike_times`` and *vice versa* if only one of the two is given.

.. note::
    The default model used for making predictions assumes that the data has been preprocessed with
    the default parameters. In general, data should undergo the same preprocessing before
    training and prediction.

Predicting spikes
-----------------

Predicting spikes is as easy as

.. code-block:: bash

    $ c2s predict data.preprocessed.pck predictions.pck

As for the preprocessing, inputs and outputs can again be MATLAB files. If the data has not been
preprocessed yet, use

.. code-block:: bash

    $ c2s predict --preprocess 1 data.pck predictions.pck

The predictions are saved in the same format as the data files, except that the entries
``spikes``, ``spike_times`` and ``calcium`` are removed to save space. By default, the prediction
uses a model which has been trained on data recorded from V1 of mice using OGB1 as indicator. But
it is possible to train a model which is better adapted to our data. Once trained, the model can be
used for prediction as follows:

.. code-block:: bash

    $ c2s predict --model model.xpck data.preprocessed.pck predictions.pck


Training a model
----------------

To train a model to fit your needs, use the command

.. code-block:: bash

    $ c2s train data.preprocessed.pck model.xpck

To print a list of available parameters to influence the training, see

.. code-block:: bash

    $ c2s train -h

Evaluation
----------

Leave-one-out cross-validation
------------------------------
