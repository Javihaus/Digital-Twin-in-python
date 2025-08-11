Quick Start Guide
=================

This guide will help you get started with the Hybrid Digital Twin framework.

Basic Usage
-----------

1. Import the necessary components:

.. code-block:: python

   from hybrid_digital_twin import HybridDigitalTwin, BatteryDataLoader

2. Load your battery data:

.. code-block:: python

   loader = BatteryDataLoader()
   data = loader.load_csv("path/to/your/data.csv")

3. Train the hybrid model:

.. code-block:: python

   twin = HybridDigitalTwin()
   metrics = twin.fit(data, target_column="Capacity")

4. Make predictions:

.. code-block:: python

   predictions = twin.predict(data)

Configuration
-------------

You can customize the model behavior with configuration:

.. code-block:: python

   config = {
       "physics_k": 0.13,
       "ml_model": {
           "hidden_layers": [64, 64],
           "learning_rate": 0.001,
           "epochs": 100
       }
   }

   twin = HybridDigitalTwin(config=config)

See the full documentation for more advanced usage patterns.