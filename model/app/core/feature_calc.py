import cupy as cp
import numpy as np
import logging

class FeatureCalculator:
    def __init__(self, config_or_sample_rate):
        if isinstance(config_or_sample_rate, dict):
            self.fs = config_or_sample_rate.get("processing", {}).get("sample_rate", 4000)
        else:
            self.fs = float(config_or_sample_rate)
        self.EPS = 1e-12
        logging.info(f"Calculating at {self.fs}Hz")
  
    def _get_signal_stats(self, data_gpu):
        mean_v = cp.mean(data_gpu, axis=0)
        abs_data = cp.abs(data_gpu)
        std_v = cp.std(data_gpu, axis=0, ddof=1)
        p2p_v = cp.max(data_gpu, axis=0) - cp.min(data_gpu, axis=0)
        # rms
        rms_v = cp.sqrt(cp.mean(cp.square(data_gpu), axis=0))  
        # kurtosis
        diff = data_gpu - mean_v
        m4 = cp.mean(cp.power(diff, 4), axis=0)
        m2 = cp.var(data_gpu, axis=0)
        kurt_v = cp.where(m2 > self.EPS, m4 / (m2**2 + self.EPS), cp.nan)

        # crest
        crest_v = cp.where(rms_v > self.EPS, cp.max(abs_data, axis=0) / (rms_v + self.EPS), cp.nan)

        # shape factor
        mean_abs_v = cp.mean(abs_data, axis=0)
        shape_v = cp.where(mean_abs_v > self.EPS, rms_v / (mean_abs_v + self.EPS), cp.nan)

        return rms_v, kurt_v, crest_v, std_v, p2p_v, shape_v

    def diagnostic_features(self, input_data):
        try:
            # push data to GPU
            data_gpu = cp.asanyarray(input_data, dtype=cp.float32)
            # calculate, vectorized
            rms, kurt, crest, std, p2p, shape = self._get_signal_stats(data_gpu)

            # stack and flatten (results grouped by axis)
            feature_matrix = cp.column_stack([rms, kurt, crest, std, p2p, shape])
            final_feature = feature_matrix.flatten()

            return final_feature
        except Exception as e:
            logging.error(f"Calculation error: {e}")
            return cp.array([cp.nan] * 18)
