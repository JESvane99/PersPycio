import lmfit

from .main_smith import MainSmith


class GaussianSmith(MainSmith):
    """
    The smith for fitting a Gaussian to data.
    """

    def __init__(self, x_data, y_data, fit_cfg):
        super().__init__(x_data, y_data, fit_cfg)

    def build_multi_gaussian(self):
        """
        building a multi-Gaussian model from the number of peaks found.
        """
        peaks, peak_properties = self.peaks
        p_window_left = peak_properties["left_bases"]
        p_window_right = peak_properties["right_bases"]

        # building the model
        model = None
        parameters = None
        for left_base, right_base, i in zip(
            p_window_left,
            p_window_right,
            range(1, len(peaks) + 1),
        ):
            _mod = lmfit.models.GaussianModel(prefix=f"Gauss{i}_")
            _mod.set_param_hint(f"Gauss{i}_amplitude", min=0)
            _params = _mod.guess(
                data=self.y_data[left_base:right_base],
                x=self.x_data[left_base:right_base],
            )
            if model is None:
                model = _mod
                parameters = _params
            else:
                model = model + _mod
                parameters += _params

        return model, parameters

    def fit(self):
        """
        Fit a Gaussian to the data.
        """
        model, parameters = self.build_multi_gaussian()
        fit_res = model.fit(data=self.y_data, x=self.x_data, params=parameters)
        return fit_res

    def write_fit_results(
        self,
        fit_res,
        path,
        fancy_name="_Multi_Gauss",
    ):
        """
        Write the fit results to a file.

        Parameters
        ----------
        fit_res : lmfit.model.ModelResult
            The results of the fit.
        path : str
            The path to write the results to.
        fancy_name : str, optional
            A fancy name for the fit.
        """
        super().write_fit_results(fit_res, path, fancy_name=fancy_name)
