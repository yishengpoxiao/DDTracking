from nibabel import streamlines
from os.path import join
import logging
from typing import Any, Tuple

from utils.utils import *
from utils.dataset import TractographyData
from model.ddtracking import DDTracking

from dipy.tracking.streamline import length
from dipy.io.utils import get_reference_info, create_tractogram_header


def calculate_high_curv(d_vector1: np.ndarray, d_vector2: np.ndarray, angle: float) -> np.ndarray:
    cos_angle = np.cos(np.radians(angle))
    cos_vects = np.clip(np.sum(d_vector1 * d_vector2, axis=-1), -1, 1)
    return cos_vects < cos_angle


class Tracker:
    def __init__(self, logger: logging.Logger = None, **args):
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger

        # if self.params['tractography_type'] not in ['deterministic', 'probabilistic']:
        #     raise ValueError(f"Invalid tractography type: {self.params['tractography_type']}, "
        #                      "type must be either 'deterministic' or 'probabilistic'")
        # self.tractography_type = self.params['tractography_type']

        self.dwi_path = self.params['dwi_path']
        self.wm_mask_path = self.params.get('wm_mask_dir', None)
        self.brain_mask_path = self.params['brain_mask_dir']
        self.track_data = None

        self.out_dir = self.params['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_name = self.params['out_name']
        self.out_tractogram_path = join(self.out_dir, self.out_name)

        self.seed_mask = None
        self.num_seeds = self.params.get('num_seeds', 1)
        self.seed_points = None
        self.track_length = None
        self.max_angle = self.params['max_angle']
        self.step_size = self.params['step_size']
        self.min_length = self.params.get('min_length', 0)
        self.max_length = self.params.get('max_length', float('inf'))
        self.track_batch_size = self.params['track_batch_size']
        
        if 'model_file' not in self.params:
            self.trained_model_dir = self.params['trained_model_dir']
            self.model_name = self.params['model_name']
            self.model_file = join(self.trained_model_dir, self.model_name + '_best.pt')
        else:
            self.model_file = self.params['model_file']

        self.num_layers = int(self.params['num_layers'])
        self.num_cells = int(self.params['num_cells'])
        self.cell_type = self.params['cell_type']
        self.dropout_prob = float(self.params['dropout_prob'])
        self.loss_type = self.params['loss_type']
        self.voxel_size = None
        self.sh_order = int(self.params.get('sh_order', 6))
        self.use_previous_dire = int(self.params['use_previous_dire'])
        self.look_ahead = bool(self.params['look_ahead'])
        self.input_type = self.params['input_type'] 
        self.grad_directions = int((self.sh_order + 1) * (self.sh_order + 2) / 2) if self.input_type in ['sh', 'fodf'] else 100

        self.model = None
        self.device = self.params['device'] if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self):
        self.logger.debug("Loading model...")
        try:
            if not os.path.exists(self.model_file):
                self.logger.error(f"Model file not found: {self.model_file}")
                raise FileNotFoundError(f"Model file not found: {self.model_file}")
            checkpoint = torch.load(self.model_file, map_location=self.device)

            self.model = DDTracking(
                input_size=self.grad_directions,
                num_layers=self.num_layers,
                num_cells=self.num_cells,
                cell_type=self.cell_type,
                dropout_rate=self.dropout_prob,
                target_dim=3,
                loss_type=self.loss_type,
                previous_dire=self.use_previous_dire,
                look_ahead=self.look_ahead,
                vox_step=self.step_size,
            ).to(self.device)

            self.model.offset = self.model.offset.to(self.device)
 
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()  # Set the model to evaluation mode
            self.logger.debug("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load the model with error: {str(e)}")

    def get_seed_mask(self) -> None:
        """
        Generates a seed mask from available WM and brain masks. Sets self.seed_mask based on available data.
        """
        if self.track_data.wm_mask is not None and self.track_data.wm_mask.size > 0:
            self.seed_mask = self.track_data.wm_mask
        else:
            self.seed_mask = self.track_data.GA > 0.18
        self.seed_mask = mask_dilate(self.seed_mask)
        self.seed_mask = np.logical_and(self.seed_mask, self.track_data.brain_mask)
    
    def _forward_tracking(self, seed_voxels: np.ndarray, volume_tensor: torch.Tensor, boundry_mask: np.ndarray) -> np.ndarray:
        batch_num_seeds = seed_voxels.shape[0]
        streamline_vox_list = [seed_voxels]
        
        streamlines_terminate = np.zeros((batch_num_seeds,), dtype=bool)
        streamlines_active = ~streamlines_terminate
        
        seed_max_steps = np.ones((batch_num_seeds,), dtype=int) * self.max_step

        hidden_state = None
        t_step = 0
        
        prev_pts = seed_voxels.copy()
        prev_direction = np.zeros((batch_num_seeds, max(self.use_previous_dire, 1), 3), dtype=np.float32)

        while True:
            with torch.no_grad():
                model_input = prev_pts
                
                if self.look_ahead:
                    # ahead_point = prev_pts + prev_direction[:, 0, :] * self.step_size
                    ahead_point = prev_pts
                    model_input = np.concatenate([model_input, ahead_point], axis=-1)
                
                if self.use_previous_dire > 0:
                    prev_dire_flat = prev_direction.reshape(prev_direction.shape[0], -1)
                    model_input = np.concatenate([model_input, prev_dire_flat], axis=-1)
                model_input_tensor = torch.as_tensor(model_input, dtype=torch.float32, device=self.device).unsqueeze(1)
                
                direction_pred, hidden_state = self.model.generate(volume_tensor, model_input_tensor, hidden_state)
                direction_pred = F.normalize(direction_pred, p=2, dim=-1).cpu().numpy()

            prev_pts += self.step_size * direction_pred

            if t_step > 0:
                angle_mask = calculate_high_curv(prev_direction[:, 0, :], direction_pred, self.max_angle)
            else:
                angle_mask = np.zeros((prev_pts.shape[0]), dtype=bool)

            if self.use_previous_dire > 1:
                prev_direction[:, 1:, :] = prev_direction[:, :-1, :]
            prev_direction[:, 0, :] = direction_pred

            EoF_mask = np.all(direction_pred == 0, axis=-1)
            inWM_mask = is_within_mask(prev_pts, boundry_mask).astype(bool)
            
            # step_length_mask = t_step >= (seed_max_steps - 1)
            
            streamlines_subset = ~streamlines_terminate
            
            # curr_terminate = angle_mask | EoF_mask | ~inWM_mask | step_length_mask
            curr_terminate = angle_mask | EoF_mask | ~inWM_mask

            streamlines_terminate[streamlines_active] |= curr_terminate
            streamlines_active = ~streamlines_terminate
            
            streamlines_subset_active = streamlines_active[streamlines_subset]

            curr_vox_nan = np.full_like(seed_voxels, np.nan)
            curr_vox_nan[streamlines_active] = prev_pts[streamlines_subset_active]
            # curr_vox_nan[streamlines_subset] = prev_pts
            streamline_vox_list.append(curr_vox_nan)

            self.logger.debug(f"Forward step {t_step}: Total seeds {batch_num_seeds}, terminated {np.sum(streamlines_terminate)}, active {np.sum(streamlines_active)}")
            
            if np.sum(streamlines_active) == 0:
                break

            prev_pts = prev_pts[streamlines_subset_active, :]
            seed_max_steps = seed_max_steps[streamlines_subset_active]
            prev_direction = prev_direction[streamlines_subset_active, ...]

            if isinstance(hidden_state, tuple):
                h_n, c_n = hidden_state
                h_n = h_n[:, streamlines_subset_active, :]
                c_n = c_n[:, streamlines_subset_active, :]
                hidden_state = (h_n, c_n)
            else:
                hidden_state = hidden_state[:, streamlines_subset_active, :]
                
            if t_step > self.max_step:
                break

            t_step += 1

        streamline_vox = np.stack(streamline_vox_list, axis=1)
        streamline_vox = np.flip(streamline_vox, axis=1)
        
        return streamline_vox
    
    def _prepare_reverse_state(self, streamline_vox: np.ndarray, volume_tensor: torch.Tensor) -> Tuple[Any, np.ndarray]:
        batch_num_seeds = streamline_vox.shape[0]
        prev_direction = np.zeros((batch_num_seeds, max(self.use_previous_dire, 1), 3), dtype=np.float32)
        hidden_state = None
        num_steps = streamline_vox.shape[1]
        
        with torch.no_grad():
            for t_idx in range(num_steps - 1):
                pts_current = streamline_vox[:, t_idx, :].copy()
                is_nan = np.isnan(pts_current).any(axis=-1)
                
                pts_current[is_nan] = 0.0
                
                model_input_tensor = torch.as_tensor(pts_current, dtype=torch.float32, device=self.device).unsqueeze(1)
                
                _, hidden_state = self.model._apply_rnn_layers(volume_tensor, model_input_tensor, hidden_state)
                
                if isinstance(hidden_state, tuple):
                    h_n, c_n = hidden_state
                    h_n[:, is_nan, :] = 0.0
                    c_n[:, is_nan, :] = 0.0
                    hidden_state = (h_n, c_n)
                else:
                    hidden_state[:, is_nan, :] = 0.0

                active_indices = np.where(~is_nan)[0]
                
                diff = streamline_vox[active_indices, t_idx+1, :] - pts_current[active_indices]
                norms = np.linalg.norm(diff, axis=-1, keepdims=True)
                prev_direction[active_indices, 0, :] = diff / (norms + 1e-6)

        return hidden_state, prev_direction

    def _reverse_tracking(self, seed_voxel: np.ndarray, volume_tensor: torch.Tensor, boundry_mask: np.ndarray,
                           hidden_state: any, prev_direction: np.ndarray, seed_max_steps: np.ndarray) -> np.ndarray:
        batch_num_seeds = seed_voxel.shape[0]
        streamline_vox_list = []
        t_step = 0
        
        prev_pts = seed_voxel.copy()
        
        streamlines_terminate = np.zeros((batch_num_seeds,), dtype=bool)
        streamlines_active = ~streamlines_terminate

        while True:
            with torch.no_grad():
                model_input = prev_pts
                
                if self.look_ahead:
                    # ahead_point = prev_pts + prev_direction[:, 0, :] * self.step_size
                    ahead_point = prev_pts
                    model_input = np.concatenate([model_input, ahead_point], axis=-1)
                
                if self.use_previous_dire > 0:
                    prev_dire_flat = prev_direction.reshape(prev_direction.shape[0], -1)
                    model_input = np.concatenate([model_input, prev_dire_flat], axis=-1)
                model_input_tensor = torch.as_tensor(model_input, dtype=torch.float32, device=self.device).unsqueeze(1)
                
                direction_pred, hidden_state = self.model.generate(volume_tensor, model_input_tensor, hidden_state)
                direction_pred = F.normalize(direction_pred, p=2, dim=-1).cpu().numpy()

            prev_pts += self.step_size * direction_pred

            if t_step > 0:
                angle_mask = calculate_high_curv(prev_direction[:, 0, :], direction_pred, self.max_angle)
            else:
                angle_mask = np.zeros((prev_pts.shape[0]), dtype=bool)

            if self.use_previous_dire > 1:
                prev_direction[:, 1:, :] = prev_direction[:, :-1, :]
            prev_direction[:, 0, :] = direction_pred

            EoF_mask = np.all(direction_pred == 0, axis=-1)
            inWM_mask = is_within_mask(prev_pts, boundry_mask).astype(bool)
            
            # step_length_mask = t_step >= (seed_max_steps)
            
            streamlines_subset = ~streamlines_terminate
            
            # curr_terminate = angle_mask | EoF_mask | ~inWM_mask | step_length_mask
            curr_terminate = angle_mask | EoF_mask | ~inWM_mask

            streamlines_terminate[streamlines_active] |= curr_terminate
            streamlines_active = ~streamlines_terminate
            
            streamlines_subset_active = streamlines_active[streamlines_subset]

            curr_vox_nan = np.full_like(seed_voxel, np.nan)
            curr_vox_nan[streamlines_active] = prev_pts[streamlines_subset_active]
            # curr_vox_nan[streamlines_subset] = prev_pts

            streamline_vox_list.append(curr_vox_nan)

            self.logger.debug(f"Reverse step {t_step}: Total seeds {batch_num_seeds}, terminated {np.sum(~streamlines_active)}, active {np.sum(streamlines_active)}")
            
            if np.sum(streamlines_active) == 0:
                break

            prev_pts = prev_pts[streamlines_subset_active, :]
            seed_max_steps = seed_max_steps[streamlines_subset_active]
            prev_direction = prev_direction[streamlines_subset_active, ...]
            if isinstance(hidden_state, tuple):
                h_n, c_n = hidden_state
                h_n = h_n[:, streamlines_subset_active, :]
                c_n = c_n[:, streamlines_subset_active, :]
                hidden_state = (h_n, c_n)
            else:
                hidden_state = hidden_state[:, streamlines_subset_active, :]
                
            if t_step > self.max_step:
                break

            t_step += 1

        streamline_vox = np.stack(streamline_vox_list, axis=1)

        return streamline_vox

    def streamline_tracking(self) -> streamlines.array_sequence.ArraySequence:
        all_streamlines = streamlines.array_sequence.ArraySequence([])

        boundry_mask = self.track_data.wm_mask if (self.track_data.wm_mask is not None and self.track_data.wm_mask.size > 0) else (self.track_data.FA > 0.15)
        boundry_mask = mask_dilate(boundry_mask)

        volume_tensor = torch.as_tensor(self.track_data.volume, dtype=torch.float32, device=self.device).unsqueeze(0)

        num_batches = (len(self.seed_points) + self.track_batch_size - 1) // self.track_batch_size
        
        for b_idx in range(num_batches):
            start_idx = b_idx * self.track_batch_size
            end_idx = min(len(self.seed_points), (b_idx + 1) * self.track_batch_size)
            
            seed_voxels = self.seed_points[start_idx:end_idx]
            self.logger.debug(f"Processing batch {b_idx+1}/{num_batches}")

            forward_streamline = self._forward_tracking(seed_voxels, volume_tensor, boundry_mask)

            rev_hidden_state, rev_prev_direction = self._prepare_reverse_state(forward_streamline, volume_tensor)

            seed_max_steps = np.ones((end_idx-start_idx,), dtype=int) * self.max_step - (~np.isnan(forward_streamline).any(axis=-1)).sum(axis=1)

            revserse_streamline = self._reverse_tracking(seed_voxels, volume_tensor, boundry_mask, rev_hidden_state, rev_prev_direction, seed_max_steps)

            streamline_voxel = np.concatenate([forward_streamline, revserse_streamline], axis=1)

            valid_mask = ~(np.isnan(streamline_voxel).any(axis=-1))
            length_counts = valid_mask.sum(axis=1)
            flat_points = streamline_voxel.reshape(-1, 3)[valid_mask.flatten()]
            batch_streamlines = np.split(flat_points, np.cumsum(length_counts)[:-1])
            
            lengths_vec = length(batch_streamlines)
            mask = np.logical_and(lengths_vec >= self.min_step, lengths_vec <= self.max_step)
            filtered_streamlines = np.array(batch_streamlines, dtype=object)[mask]
            
            all_streamlines.extend(filtered_streamlines)
            
        return all_streamlines

    def track(self) -> streamlines.tractogram.Tractogram:
        self.track_data = TractographyData('track', self.dwi_path, self.brain_mask_path, self.wm_mask_path, sh_order=self.sh_order, input_type=self.input_type)
        
        self.get_seed_mask()
        self.seed_points = init_seeds_in_voxel(self.seed_mask, self.num_seeds)
        self.voxel_size = np.mean(np.abs(self.track_data.dwi.affine)[np.diag_indices(4)][:3])
        
        self.max_step = self.max_length / self.voxel_size
        self.min_step = self.min_length / self.voxel_size
        
        out_streamlines = self.streamline_tracking()
        out_tractogram = streamlines.tractogram.Tractogram(streamlines=out_streamlines)
        
        out_tractogram.affine_to_rasmm = self.track_data.dwi.affine
        filetype = streamlines.detect_format(self.out_tractogram_path)
        reference = get_reference_info(self.dwi_path)
        header = create_tractogram_header(filetype, *reference)

        streamlines.save(out_tractogram, self.out_tractogram_path, header=header)
        return out_tractogram
