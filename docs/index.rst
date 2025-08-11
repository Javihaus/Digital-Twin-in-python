Hybrid Digital Twin Documentation
================================

Welcome to the Hybrid Digital Twin framework documentation!

This framework implements a professional-grade hybrid approach for Li-ion battery modeling,
combining physics-based modeling with machine learning for superior accuracy and reliability.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
   examples

Quick Start
===========

To get started with the Hybrid Digital Twin framework:

.. code-block:: python

   from hybrid_digital_twin import HybridDigitalTwin, BatteryDataLoader

   # Load data
   loader = BatteryDataLoader()
   data = loader.load_csv("discharge.csv")

   # Train model
   twin = HybridDigitalTwin()
   metrics = twin.fit(data)

   # Make predictions
   predictions = twin.predict(data)

Installation
============

Install from PyPI:

.. code-block:: bash

   pip install hybrid-digital-twin

Or for development:

.. code-block:: bash

   git clone https://github.com/Javihaus/Digital-Twin-in-python.git
   cd Digital-Twin-in-python
   pip install -e ".[dev]"

API Reference
=============

.. toctree::
   :maxdepth: 2

   api/core
   api/models
   api/data
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`