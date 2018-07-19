.. Starfish documentation master file, created by
   sphinx-quickstart on Wed Mar 28 13:13:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API documentation
=================

Starfish is organized into pipeline stages.

Registration
------------

Registration is handled by the :py:func:`starfish.pipeline.registration.Registration`.  The individual algorithms that
implement registration must extend :py:func:`starfish.pipeline.registration`RegistrationAlgorithmBase`.

.. autoclass:: starfish.pipeline.registration.Registration
   :members:
   :inherited-members:
   :exclude-members: add_to_parser, algorithm_to_class_map

.. autoclass:: starfish.pipeline.registration.RegistrationAlgorithmBase
   :members:

Registration Algorithms
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: starfish.pipeline.registration.fourier_shift.FourierShiftRegistration
   :members:
   :exclude-members: add_arguments, from_cli_args, get_algorithm_name

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
