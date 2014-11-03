Installation
============

Besides Python, `NumPy <http://www.scipy.org>`_, and `Scipy <http://www.scipy.org>`_,
the c2s package depends on the `Conditional Modeling Toolkit <https://github.com/lucastheis/cmt>`_.
Note that as it currently stands, this toolkit works best under Linux and is hard to get to work under
Windows. After installing the toolkit, the c2s package can be installed by running the following on the
command line:

.. code-block:: bash

    $ pip install git+https://github.com/lucastheis/cmt.git

After installation, the following command should be available and output some help:

.. code-block:: bash

    $ c2s -h
