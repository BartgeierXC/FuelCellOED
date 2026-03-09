import numpy as np
from scipy.differentiate import derivative as der
from src.math_utils.derivatives.interface.derivative_calculator import DerivativeCalculator
from src.math_utils.scaler.interface.scaler import ParameterScaler
from src.model.interface.fuel_cell_stack_model import FuelCellStackModel


class NumericDerivativeCalculator(DerivativeCalculator):
    """
    A class for calculating numerical derivatives using finite difference methods.
    This class inherits from the DerivativeCalculator base class.
    """

    def __init__(self, model: FuelCellStackModel, scaler: ParameterScaler):
        super().__init__()
        self.model = model
        self.scaler = scaler

    def calculate_derivative(self, data, variable, i):
        """
        Calculates the numerical derivative of the given data with respect to the specified variable.

        Supports both scalar operating points and polarization curves (array of currents).

        :param data: The data for which the derivative is to be calculated. Can be 2D array where:
                     - Each row is an operating point, e.g. [p_K, T_S, stoic, RH, I_Z]
                     - I_Z can be a scalar OR an array (polarization curve)
        :param variable: The parameter vector (theta) with respect to which the derivative is calculated.
        :param i: The index of the parameter in the variable vector.
        :return: 1D numpy array of derivatives. If data contains polarization curves, the array
                 contains all derivatives (one per measurement point across all curves).
        """

        return self._calculate_derivatives_num(
            x_k=data,
            theta=variable,
            i=i
        )

    def _calculate_derivatives_num(self,
                                   x_k: np.ndarray,
                                   theta: np.ndarray,
                                   i: int) -> np.ndarray:
        """
        Computes numerical derivatives using finite differences.

        Handles both scalar and vectorized inputs:
        - Scalar: x0[4] is a single current value → returns one derivative per operating point
        - Vectorized (polarization curve): x0[4] is array of currents → returns multiple derivatives per operating point

        Mathematical formula (Fisher Information Matrix):
        FIM_ij = (1/σ²) * Σ_l [∂f/∂θᵢ(xₗ) * ∂f/∂θⱼ(xₗ)]
        where l runs over ALL measurement points (across all curves if polarization curves are used).
        """
        y_der_vals = []
        base_x = np.array(theta)

        for x0 in x_k:
            def fc_model(input_val):
                input_val = np.atleast_1d(
                    input_val).flatten()  # needed, as input val is only scalar in first iteration of n in derivative
                results = []
                for val in input_val:
                    x_temp_i = base_x.copy()
                    x_temp_i[i] = val
                    result = self.model(scaler = self.scaler,theta=x_temp_i.tolist(), x=x0)
                    results.append(float(result) if np.isscalar(result) else result)
                return np.array(results)

            du_res = der(fc_model, theta[i],
                         preserve_shape=True,
                         order=2,
                         maxiter=20,
                         step_direction=1,
                         initial_step=0.5,
                         tolerances={"atol": 0.001})

            # Handle both scalar and array derivatives (for polarization curves)
            # du_res.df can be a scalar or array depending on whether x0 contains a polarization curve
            if isinstance(du_res.df, np.ndarray):
                # Polarization curve case: flatten all derivative values into the list
                y_der_vals.extend(du_res.df.flatten())
            else:
                # Scalar case: append the single value
                y_der_vals.append(du_res.df)

            # du_res = scalers[i].inverse_transform(du_res.df.reshape(-1, 1)).flatten()
            # print(f"resulting value for derivative {i}: {du_res.df}")
        return np.array(y_der_vals)