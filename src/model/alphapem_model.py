from abc import ABC

import numpy as np

from src.model.interface.fuel_cell_stack_model import FuelCellStackModel
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.parameter_set.interface.parameter_set import ParameterSet

from alphapem import AlphaPEM


class AlphaPEMStackModel(AlphaPEM, FuelCellStackModel):
    """
    Base class for fuel cell stack models.
    This class should be inherited by specific fuel cell stack model implementations.
    """

    def __init__(self, parameter_set: ParameterSet = None):
        """
        Initializes the HahnStackModel with a given parameter set.

        :param parameter_set: An instance of ParameterSet containing the parameters for the model.
        """
        super().__init__(accessible_physical_parameters, undetermined_physical_parameters, model_parameters)
        if parameter_set is None:
            parameter_set = ParameterSet()  # use the default parameter set
        self.parameter_set = parameter_set

    def simulate_model(self, operating_inputs, simulation_parameters: ParameterSet = None):
        """
        Simulates AlphaPEM fuel cell stack model with the given parameters.

        :param operating_inputs: A numpy array containing the reduced operating conditions for the simulation.
        :param simulation_parameters: An optional ParameterSet containing the parameters for the simulation.
        :return: The results of the simulation (ie: voltage).
        """

        super().simulate_model(operating_inputs, current_parameters, computing_parameters)

        return {'U': self.variables['Ucell']} # This should be adjusted as it returns the cell voltage function of time, not current density.


    def simulation_wrapper(self, scaler: ParameterScaler, x: np.ndarray, theta: np.ndarray,
                           full_output: bool = False):
        """
        Wrapper for the simulation method that handles scaling of parameters.
        :param scaler: The scaler to be used for scaling parameters.
        :param theta: The parameters to be scaled and used in the simulation.
        :param x: Additional data for the simulation.
        :param full_output: If True, returns additional output data.
        :return: The results of the simulation, potentially including additional output data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self, *args, **kwargs):
        return self.simulation_wrapper(*args, **kwargs)
