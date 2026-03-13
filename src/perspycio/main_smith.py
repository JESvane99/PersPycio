"""
A smithing school for the Fitting module.
When fitting armor to a knight reaper, the smith must be precise and careful.
(Or else the knight reaper will be very uncomfortable.)

Helper module for fitting a function to data.
"""

import h5py
import numpy as np
import uncertainties
from scipy.signal import find_peaks


class MainSmith:
    def __init__(self, x_data, y_data, fit_cfg):
        self.cfg = fit_cfg
        self.y_data = y_data
        self.x_data = x_data[: len(y_data)]

    @property
    def peaks(self):
        """
        Find peaks in the data.

        Parameters
        ----------
        y_data : np.ndarray
            The data to find peaks in.
        fit_cfg : FittingCfg

        Returns
        -------
        peaks : np.ndarray
            The indices of the peaks.
        peak_properties : dict
            Properties of the peaks.
        """
        fit_cfg = self.cfg
        n_models = fit_cfg.max_n_models

        # preferences for peak finding
        height = fit_cfg.min_peak_height
        vert_distance = fit_cfg.y_distance_thresh
        hor_distance = fit_cfg.x_distance_thresh
        prominence = fit_cfg.peak_prominence
        width = fit_cfg.peak_width
        wlen = fit_cfg.wlen
        rel_height = fit_cfg.rel_height
        plateau_size = fit_cfg.plateau_size

        # finding peaks
        peaks, peak_properties = find_peaks(
            self.y_data,
            height=height,
            threshold=vert_distance,
            distance=hor_distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            rel_height=rel_height,
            plateau_size=plateau_size,
        )

        # iterate until number of peaks matches max number of models
        while len(peaks) > n_models:
            hor_distance += fit_cfg.peak_distance_increase
            peaks, peak_properties = find_peaks(
                self.y_data,
                height=height,
                threshold=vert_distance,
                distance=hor_distance,
                prominence=prominence,
                width=width,
                wlen=wlen,
                rel_height=rel_height,
                plateau_size=plateau_size,
            )

        return peaks, peak_properties

    def write_fit_results(
        self,
        fit_res,
        path,
        fancy_name=None,
    ):
        """
        Write the fit results to a file.

        Parameters
        ----------
        fit_res : lmfit.model.ModelResult
            The results of the fit.
        path : pathlib.Path
            The path of the file to write to.
        fancy_name : str, optional
            A fancy name to add to the file name post-fix, by default None
        """
        uvars = fit_res.uvars if hasattr(fit_res, "uvars") else "No uncertainties found"
        cor_mode = "table" if hasattr(fit_res, "uvars") else "list"
        peaks, peak_properties = self.peaks

        write_map = {
            "fit_data": fit_res.best_fit,
            "fit_report": fit_res.fit_report(correl_mode=cor_mode),
            "best_values": fit_res.best_values,
            "uncertainties": uvars,
            "correl_mode": cor_mode,
            "peaks": peaks,
            "peak_properties": peak_properties,
        }
        if fancy_name:
            path = path.with_stem(path.stem + fancy_name)
        with h5py.File(path, "w") as file:
            for k, v in write_map.items():
                if "data" in k:
                    file.create_dataset(k, data=v, dtype=np.min_scalar_type(v))
                else:
                    meta = file.require_group(name="META")
                    if isinstance(v, dict):
                        branch = meta.require_group(name=k)
                        for key, value in v.items():
                            if isinstance(value, uncertainties.UFloat):
                                _val = uncertainties.std_dev(value)
                            else:
                                _val = value
                            branch.create_dataset(
                                name=key, data=_val, dtype=np.min_scalar_type(_val)
                            )
                    else:
                        meta.create_dataset(name=k, data=v)
