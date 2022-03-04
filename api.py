"""
Online application to access the trained neural network.

Author: Gijs G. Hendrickx
"""
import sys
import os


def run():
    """Run web-API locally."""
    from neural_network.application.app import APP
    APP.run_server()


# add root-directory to PYTHONPATH
root = os.sep.join(os.path.abspath(os.path.dirname(__file__)).split(os.sep)[:-1])
if root not in sys.path:
    sys.path.append(root)

# run web-API
run()
