"""Learning port-Hamiltonian dynamics from data.

The classes import torch lazily (only when instantiated), so importing this
package does not require the ``[torch]`` extra — constructing a
:class:`PortHamiltonianNN` does.
"""

from otwin.learn.phnn import PortHamiltonianNN

__all__ = ["PortHamiltonianNN"]
