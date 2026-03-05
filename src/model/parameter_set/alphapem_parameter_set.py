from src.model.parameter_set.interface.parameter_set import ParameterSet

from alphapem.config.parameters import calculate_physical_parameters, calculate_computing_parameters


class AlphaPEMParameterSet(ParameterSet):
    """
    A class representing the parameter set for the AlphaPEM fuel cell model.
    It extends the ParameterSet class to include specific parameters for the AlphaPEM model.
    """

    def __init__(self, free_parameters: list[tuple[str, float]] = None,
                 values: list[float] = None):
        """
        Initializes the AlphaPEMParameterSet with cell and free parameters depending on theta.

        :param free_parameters: A dict containing the free parameters for the model.
        """
        super().__init__()

        # Model configuration
        type_fuel_cell = "ZSW-GenStack"
        type_current = "polarization"
        voltage_zone = "full"
        type_auxiliary = "no_auxiliary"
        type_purge = "no_purge"
        type_display = "no_display"
        type_plot = "fixed"

        #   Physical parameters
        (Hacl, Hccl, Hmem, Hgdl, epsilon_gdl, epsilon_c, Hmpl, epsilon_mpl, Hagc, Hcgc, Wagc, Wcgc, Lgc,
         nb_channel_in_gc, Ldist, Lm, A_T_a, A_T_c, Vasm, Vcsm, Vaem, Vcem, Aact, nb_cell, e, K_l_ads, K_O2_ad_Pt, Re,
         i0_c_ref, kappa_co, kappa_c, C_scl) = calculate_physical_parameters(type_fuel_cell)
        #   Computing parameters
        nb_gc, nb_gdl, nb_mpl, t_purge, rtol, atol = calculate_computing_parameters()

        cell_parameters = {
            # Accessible physical parameters
            'Aact': Aact,
            'nb_cell': nb_cell,
            'Hagc': Hagc,
            'Hcgc': Hcgc,
            'Wagc': Wagc,
            'Wcgc': Wcgc,
            'Lgc': Lgc,
            'epsilon_mpl': epsilon_mpl,
            'epsilon_c': epsilon_c,
            'C_scl': C_scl,
            'nb_channel_in_gc': nb_channel_in_gc,
            'Ldist': Ldist,
            'Lm': Lm,
            'A_T_a': A_T_a,
            'A_T_c': A_T_c,
            'Vasm': Vasm,
            'Vcsm': Vcsm,
            'Vaem': Vaem,
            'Vcem': Vcem,

            # Model parameters
            'nb_gc': nb_gc,
            'nb_gdl': nb_gdl,
            'nb_mpl': nb_mpl,
            't_purge': t_purge,
            'rtol': rtol,
            'atol': atol,

            # Computing parameters
            'type_fuel_cell': type_fuel_cell,
            'type_current': type_current,
            'voltage_zone': voltage_zone,
            'type_auxiliary': type_auxiliary,
            'type_purge': type_purge,
            'type_display': type_display,
            'type_plot': type_plot,
        }

        self.data['cell_parameters'] = cell_parameters

        # Undetermined physical parameters
        default_free_parameters = {
            'Hgdl': Hgdl,
            'Hmpl': Hmpl,
            'Hmem': Hmem,
            'Hacl': Hacl,
            'Hccl': Hccl,
            'epsilon_gdl': epsilon_gdl,
            'e': e,
            'K_l_ads': K_l_ads,
            'K_O2_ad_Pt': K_O2_ad_Pt,
            'Re': Re,
            'i0_c_ref': i0_c_ref,
            'kappa_co': kappa_co,
            'kappa_c': kappa_c,
        }

        # --- New unified logic ---
        # self.modify_free_parameters = []

        if free_parameters is None:
            # No modification — use defaults
            pass

        elif isinstance(free_parameters, dict):
            self.modify_free_parameters = []
            # Dict input — direct mapping
            for param, value in free_parameters.items():
                if param not in default_free_parameters:
                    raise KeyError(f"Parameter '{param}' not recognized.")
                default_free_parameters[param] = value
                self.modify_free_parameters.append(param)

        elif isinstance(free_parameters, (list, tuple)) and values is not None:
            self.modify_free_parameters = []
            # List of names + separate array of values
            if len(free_parameters) != len(values):
                raise ValueError("Number of parameter names and values must match.")
            for name, val in zip(free_parameters, values):
                if name not in default_free_parameters:
                    raise KeyError(f"Parameter '{name}' not recognized.")
                default_free_parameters[name] = val
                self.modify_free_parameters.append(name)

        elif isinstance(free_parameters, (list, tuple)) and values is None:
            self.modify_free_parameters = []
            for name in free_parameters:
                if name not in default_free_parameters:
                    raise KeyError(f"Parameter '{name}' not found in default_free_parameters.")
                self.modify_free_parameters.append(name)

        else:
            raise TypeError(
                "free_parameters must be a dict, or a list of names with a matching 'values' argument."
            )

        self.data['free_parameters'] = default_free_parameters
