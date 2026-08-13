#! python3
# -*- encoding: utf-8 -*-

import numpy as np

try:
    import torch
except ImportError:
    torch = None

TORCH_AVAILABLE = torch is not None
CUDA_AVAILABLE = bool(
    TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 0
)

from .simulator_utils import np_event_generation
if TORCH_AVAILABLE:
    from .simulator_utils import event_generation_torch

class EventSim(object):
    def __init__(
            self,
            cfg,
            output_folder: str = '',
            video_name: str = '',
            sim_backend: str = 'auto',
    ):
        """
        Parameters
        ----------
        cfg: config
        output_folder: str
            folder of output data file
        video_name: str
            name of input video / output data file
        """

        # set parameters in model
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 \
            = cfg.SENSOR.K[0], cfg.SENSOR.K[1], cfg.SENSOR.K[2], cfg.SENSOR.K[3], cfg.SENSOR.K[4], cfg.SENSOR.K[5]
        # K is 8-dim [k1..k6, k_on, k_off]; a 6-dim K defaults k_on = k_off = 1.0.
        self.k_on = float(cfg.SENSOR.K[6]) if len(cfg.SENSOR.K) >= 7 else 1.0
        self.k_off = float(cfg.SENSOR.K[7]) if len(cfg.SENSOR.K) >= 8 else 1.0

        self.sim_backend = self._resolve_backend(sim_backend)
        self.device = None
        if self.sim_backend == 'cuda':
            self.device = torch.device('cuda')
        elif self.sim_backend == 'cpu':
            self.device = torch.device('cpu')

        # output file
        # path = os.path.join(output_folder, video_name + '.npy')

        # init
        self.reset()

    @staticmethod
    def _resolve_backend(sim_backend: str) -> str:
        backend = (sim_backend or 'auto').lower()
        valid = {'auto', 'cuda', 'cpu', 'numpy'}
        if backend not in valid:
            raise ValueError(f"Unknown sim_backend='{sim_backend}', expected one of {sorted(valid)}")

        if backend == 'auto':
            if CUDA_AVAILABLE:
                return 'cuda'
            if TORCH_AVAILABLE:
                return 'cpu'
            return 'numpy'

        if backend == 'cuda':
            if not TORCH_AVAILABLE:
                raise RuntimeError("sim_backend='cuda' requested but PyTorch is not installed.")
            if not CUDA_AVAILABLE:
                raise RuntimeError("sim_backend='cuda' requested but CUDA is not available.")
            return 'cuda'

        if backend == 'cpu':
            if not TORCH_AVAILABLE:
                raise RuntimeError("sim_backend='cpu' requested but PyTorch is not installed.")
            return 'cpu'

        # backend == 'numpy'
        return 'numpy'

    def reset(self):
        '''
            resets so that next use will reinitialize the base frame
        '''
        self.baseFrame = None
        self.t_previous = None  # time of previous frame

    def generate_events(
            self, new_frame: np.ndarray,
            t_frame: int) -> np.ndarray:
        """
        Notes:
            Compute events in new frame.

        Parameters
            new_frame: np.ndarray
                [height, width]
            t_frame: int
                timestamp of new frame in us (1e6)

        Returns
            events: np.ndarray if any events, else None
                [N, 4], each row contains [timestamp (us), x cordinate, y cordinate, sign of event].
        """
        if self.sim_backend in {'cuda', 'cpu'}:
            if new_frame.dtype == np.uint16:
                # Some torch builds reject uint16 NumPy arrays in from_numpy.
                new_frame = new_frame.astype(np.int32, copy=False)
            new_frame = torch.from_numpy(new_frame).to(device=self.device, dtype=torch.float64)
            t_frame = float(t_frame)
            # ------------------
            # Initialization
            if self.baseFrame is None:
                self.baseFrame = new_frame
                self.t_previous = t_frame
                self.delta_vd_res = torch.zeros_like(new_frame)  # initialize residual voltage change $\Delta V_d^{res}$
                self.t_now = torch.ones_like(new_frame, dtype=torch.float64) * self.t_previous
                # Per-polarity thresholds; k_on = k_off = 1.0 is the unweighted case.
                # torch.ones_like(...) byte-for-byte.
                self.thres_off = torch.full_like(new_frame, self.k_off)
                self.thres_on = torch.full_like(new_frame, self.k_on)
                return None

            if t_frame <= self.t_previous:
                raise ValueError("this frame time={} must be later than previous frame time={}".format(t_frame, self.t_previous))
            # ------------------
            # Calculte distribution parameters of Brownian Motion with Drift in Eq. (10)(11)
            delta_light = (new_frame - self.baseFrame)  # delta L
            avg_light = (new_frame + self.baseFrame) / 2.0  # average L
            denominator = 1 / (avg_light + self.k2)
            mu_clean = (self.k1 * delta_light / (t_frame - self.t_previous)) * denominator
            mu = mu_clean + self.k4 + self.k5 * avg_light
            var_clean = (self.k3 * torch.sqrt(avg_light)) * denominator
            var = var_clean + self.k6
            ori_shape = mu.shape
            # ------------------
            # Event Generation!
            e_t, e_x, e_y, e_p, e_dvd = event_generation_torch(
                self.thres_on, self.thres_off,
                mu, var,
                self.delta_vd_res, self.t_now, t_frame
            )
            if e_t.shape[0] > 0:
                e_t = torch.round(e_t).int()
                event_tensor = torch.stack([e_t, e_x, e_y, e_p], dim=1)
                _, sorted_idx = torch.sort(e_t)
                event_tensor = event_tensor[sorted_idx, :]
                event_tensor = event_tensor.contiguous().to('cpu').numpy().astype(np.int32)
            else:
                event_tensor = None

            # Update
            self.delta_vd_res = e_dvd.reshape(ori_shape)
            self.t_now = torch.ones_like(self.t_now, device=self.t_now.device) * t_frame
            self.t_previous = t_frame
            self.baseFrame = new_frame
            return event_tensor
        else:
            new_frame = new_frame.astype(np.float64)
            t_frame = float(t_frame)
            # ------------------
            # Initialization
            if self.baseFrame is None:
                self.baseFrame = new_frame
                self.t_previous = t_frame
                self.delta_vd_res = np.zeros_like(new_frame)
                self.t_now = np.ones_like(new_frame, dtype=np.float32) * self.t_previous
                # Per-polarity thresholds; k_on = k_off = 1.0 is the unweighted case.
                # np.ones_like(...) byte-for-byte.
                self.thres_off = np.full_like(new_frame, self.k_off)
                self.thres_on = np.full_like(new_frame, self.k_on)
                return None

            if t_frame <= self.t_previous:
                raise ValueError("this frame time={} must be later than previous frame time={}".format(t_frame, self.t_previous))
            # ------------------
            # Calculte distribution parameters of Brownian Motion with Drift in Eq. (10)(11)
            delta_light = (new_frame - self.baseFrame)  # delta L
            avg_light = (new_frame + self.baseFrame) / 2.0  # average L
            denominator = 1 / (avg_light + self.k2)
            mu_clean = (self.k1 * delta_light / (t_frame - self.t_previous)) * denominator
            mu = mu_clean + self.k4 + self.k5 * avg_light
            var_clean = (self.k3 * np.sqrt(avg_light)) * denominator
            var = var_clean + self.k6
            ori_shape = mu.shape
            # ------------------
            # Event Generation!
            e_t, e_x, e_y, e_p, e_dvd = np_event_generation(
                self.thres_on, self.thres_off,
                mu, var,
                self.delta_vd_res, self.t_now, t_frame
            )
            if e_t.shape[0] > 0:
                e_t = np.round(e_t).astype(np.int32)
                event_tensor = np.stack([e_t, e_x, e_y, e_p], axis=1)
                sorted_idx = np.argsort(e_t)
                event_tensor = event_tensor[sorted_idx, :].astype(np.int32)
            else:
                event_tensor = None

            # Update
            self.delta_vd_res = e_dvd.reshape(ori_shape)
            self.t_now = np.ones_like(self.t_now) * t_frame
            self.t_previous = t_frame
            self.baseFrame = new_frame
            return event_tensor
