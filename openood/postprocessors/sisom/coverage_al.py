"""Coverage-based Active Learning and OOD Detection.

This module provides the DSC (Distance-based Surprise Coverage) class
used for both Out-of-Distribution detection and Active Learning sample selection.

OOD Detection Usage (via SisomPostprocessor):
    The SisomPostprocessor uses DSC.calculate_ood() to compute coverage-based
    anomaly scores. See sisom_postprocessor.py for the full pipeline.

Active Learning Usage:
    The DSC class also supports active learning via its top_k() method:

        from openood.postprocessors.sisom.coverage_al import DSC
        from openood.postprocessors.sisom import tool

        # 1. Get layer sizes from model
        layer_size_dict = tool.get_layer_output_sizes(model, sample_input, mode="nac")

        # 2. Initialize DSC coverage
        dsc = DSC(model, layer_size_dict, hyper=0.001, dataset="cifar10",
                  sigmoids={"AdaptiveAvgPool2d-1": 100, "Sequential-3": 1000})

        # 3. Build coverage from labeled training data
        dsc.use = "AL"
        dsc.build(train_loader)  # train_loader yields (data, label) tuples

        # 4. Select top-k most informative samples
        gains, indices = dsc.top_k(unlabeled_loader, k=100,
                                    coverage_al="optimal2")
        # indices contains the most informative sample indices to label next
"""

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from . import tool


class DSC:
    """Distance-based Surprise Coverage for OOD detection and Active Learning.

    Computes coverage-based anomaly scores using distance ratios in the
    neural activation space. Supports both OOD inference (via calculate_ood/check)
    and active learning sample selection (via top_k).
    """

    def __init__(self, model, layer_size_dict, cycle_num=0, hyper=None, dataset="cifar10", **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset
        self.layer_size_dict = layer_size_dict
        self.cycle_num = cycle_num
        self.sigmoids = kwargs.get("sigmoids", None)
        kwargs.pop('sigmoids', None)
        self._init_variable(hyper, **kwargs)

    def _init_variable(self, hyper, min_var=1e-5, num_class=10):
        self.name = 'DSC'
        assert hyper is not None
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.use = "AL"
        self.data_count = 0
        self.weight = 0
        self.weight_al = (0, 0)
        self.std = (1, 1)
        self.current = 0
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.use_nac = True
        if self.use_nac:
            self.min_var = 0
        self.plot_activation_histogram = False
        self.mean_dict = {}
        self.var_dict = {}
        self.layer_cache = {}
        self.kde_cache = {}
        self.SA_cache = {}
        self.layer_selection = None
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, data_loader):
        """Build the coverage model from training data."""
        print('Building Mean & Var...')
        if self.min_var > 0:
            for i, (data, label) in enumerate(tqdm(data_loader)):
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                self._set_mean_var(data, label)
        self._set_mask()
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self._build_SA(data, label)
        self._to_numpy()

    def _set_mean_var(self, data, label):
        batch_size = label.size(0)
        if self.use_nac:
            layer_output_dict, score = tool.get_layer_output_nac(self.model, data, layer_selection=self.layer_selection,
                                                                  use=self.use, dataset=self.dataset,
                                                                  sigmoids=self.sigmoids)
        else:
            layer_output_dict = tool.get_layer_output(self.model, data, layer_selection=self.layer_selection)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                self.data_count += batch_size
                self.mean_dict[layer_name] = ((self.data_count - batch_size) * self.mean_dict[
                    layer_name] + layer_output.sum(0)) / self.data_count
                self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
                                            + (self.data_count - batch_size) * (
                                                        (layer_output - self.mean_dict[layer_name]) ** 2).sum(
                    0) / self.data_count ** 2

    def _set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def _build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        if self.use_nac:
            layer_output_dict, score = tool.get_layer_output_nac(self.model, data_batch,
                                                                  layer_selection=self.layer_selection, use=self.use,
                                                                  dataset=self.dataset, sigmoids=self.sigmoids)
        else:
            layer_output_dict = tool.get_layer_output(self.model, data_batch, layer_selection=self.layer_selection)

        if self.plot_activation_histogram:
            for key, activations in layer_output_dict.items():
                if key not in self.layer_cache:
                    self.layer_cache[key] = activations.detach().cpu()
                else:
                    self.layer_cache[key] = torch.cat((self.layer_cache[key], activations.detach().cpu()), dim=0)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        valid_mask = ~(torch.any(SA_batch.isnan(), dim=1) | torch.any(SA_batch.isinf(), dim=1))
        SA_batch = SA_batch[valid_mask]
        label_batch = label_batch[valid_mask]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]

    def _to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    # ── OOD Inference ──────────────────────────────────────────────────

    def check(self, data, label, gain_mode="standard", surprise_strategy="coverage", coverage_method="NLC",
              use_prediction=False):
        """Run OOD check on a batch, returning gain, energy, layer outputs, and distances."""
        total, energy, layer_output_dict, distadistb = self.calculate_ood(data, label,
                                                                           surprise_strategy=surprise_strategy,
                                                                           use_prediction=use_prediction)
        return total, energy, layer_output_dict, distadistb

    def calculate_ood(self, data_batch, label_batch, surprise_strategy="coverage", use_prediction=False):
        """Compute DSC-based OOD scores for a batch."""
        cove_set = set()
        dsa_list = []
        energy_list = []
        dista_list = []
        distb_list = []
        SA_batch = []
        layer_output_dict, score = tool.get_layer_output_nac(self.model, data_batch,
                                                              layer_selection=self.layer_selection, use=self.use,
                                                              dataset=self.dataset, sigmoids=self.sigmoids)
        if label_batch is not None:
            batch_size = label_batch.size(0)
        else:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = 1
        if use_prediction:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = len(label_batch)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))

        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()  # [batch_size, num_neuron]
        energy = tool.get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            if int(label.cpu()) not in self.SA_cache:
                dsa_list.append(0.0)
                energy_list.append(energy[i])
                dista_list.append(0.0)
                distb_list.append(0.0)
                continue

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            if self.weight == 0 and len(dist_a_list) >= 2:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            if surprise_strategy == "dist_a":
                dsa_list.append(float(dist_a))
            elif surprise_strategy == "dist_b":
                if dist_b > 0:
                    db = 1 / dist_b
                else:
                    db = 1e-6
                dsa_list.append(float(db))
            else:
                dsa_list.append(dsa)
                energy_list.append(energy[i])
                dista_list.append(float(dist_a))
                distb_list.append(float(dist_b))
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        if surprise_strategy == "coverage":
            return cove_set, layer_output_dict
        else:
            if batch_size == 1:
                if surprise_strategy == "dist_a":
                    return dist_a, layer_output_dict
                elif surprise_strategy == "dist_b":
                    return db, layer_output_dict
                else:
                    return dsa, layer_output_dict
            else:
                return torch.tensor(dsa_list).to(self.device), torch.tensor(energy_list).to(
                    self.device), layer_output_dict, (dista_list, distb_list)

    def assess_ood(self, data_loader, coverage_method="NC", surprise_strategy="coverage"):
        """Assess OOD data using the coverage method.

        Args:
            data_loader: DataLoader yielding dicts with 'data' key
            coverage_method: Coverage method name (e.g., "DSC")
            surprise_strategy: Strategy for surprise computation

        Returns:
            Tensor of gain values for each sample
        """
        gain_list = torch.tensor([])
        with torch.no_grad():
            for b in tqdm(data_loader):
                data = b["data"]
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                if "SC" not in coverage_method:
                    raw_gain = self._step_base(data).unsqueeze(0).cpu()
                else:
                    label = b["label"]
                    raw_gain, _ = self.step(data, label, surprise_strategy=surprise_strategy)
                gain_list = torch.cat((gain_list.cpu(), raw_gain))
        return gain_list

    # ── Active Learning ────────────────────────────────────────────────

    def assess(self, data_loader, gain_mode="standard", surprise_strategy="surprise"):
        """Assess all samples in the data loader, returning (gain, energy) pairs."""
        if "energy" in gain_mode:
            surprise_strategy = "surprise"
        gain_list = torch.tensor([], device="cpu")
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            dsa, energy = self.step(data, label, surprise_strategy=surprise_strategy)
            gains = torch.cat([dsa.cpu().unsqueeze(1), energy.cpu().unsqueeze(1)], dim=1)
            gain_list = torch.cat([gain_list.cpu(), gains.cpu()], dim=0)
        return gain_list

    def step(self, data, label, gain_mode="standard", surprise_strategy="coverage", coverage_al="incremental"):
        """Compute DSA and energy for a batch (AL mode)."""
        dsa, energy, layer_output_dict = self.calculate(data, label, surprise_strategy)
        return dsa, energy

    def _step_base(self, data):
        """Base step for non-SC coverage methods (used by assess_ood)."""
        cove_dict, layer_output_dict = self.calculate(data, None)
        gain = self._gain_base(cove_dict)
        if gain is not None:
            self._update_base(cove_dict, gain)
            if isinstance(gain, tuple):
                return gain[0]
            else:
                return gain
        else:
            return torch.tensor(0)

    def top_k(self, data_loader, k, coverage_al, no_label=False, gain_mode="standard", surprise_strategy="coverage"):
        """Select top-k most informative samples for active learning."""
        indices_and_gains = []
        cached_SA_batches, cached_SA_labels, cached_SA_scores = self._build_greedy(data_loader)
        cached_data = (cached_SA_batches, cached_SA_labels, cached_SA_scores)
        if coverage_al == "optimal":
            for _ in tqdm(range(k)):
                top_sample = self._top_k_loop(indices_and_gains, data_loader, coverage_al, query_size=k,
                                              cached_data=cached_data)
                indices_and_gains.append(top_sample)
        elif coverage_al == "optimal2":
            indices_and_gains = self._top_k_oneshot(indices_and_gains, data_loader, coverage_al, query_size=k,
                                                    cached_data=cached_data)
        elif coverage_al == "individual" or coverage_al == "incremental":
            indices_and_gains = self._top_k_loop(indices_and_gains, data_loader, coverage_al, no_label=no_label,
                                                 gain_mode=gain_mode, surprise_strategy=surprise_strategy, query_size=k,
                                                 cached_data=cached_data)

        indices_and_gains.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _, _ in indices_and_gains]
        top_gains = [gain.cpu().item() if torch.is_tensor(gain) else gain for _, gain, _ in indices_and_gains]
        top_energies = [energy.cpu().item() if torch.is_tensor(energy) else energy for _, _, energy in
                        indices_and_gains]
        return top_gains, top_indices

    def _top_k_loop(self, indices_and_gains, data_loader, coverage_al, no_label=False,
                    gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        max_sample = (-1, -1.0, -1)
        selected_indices = [idx for idx, _, _ in indices_and_gains]
        for i in tqdm(range(cached_data[0].shape[0])):
            sample_idx = i
            if coverage_al == "incremental":
                dsa, energy = self.calculate_greedy(cached_data[0][sample_idx][None, :],
                                                     cached_data[1][sample_idx].unsqueeze(0),
                                                     cached_data[2][sample_idx].unsqueeze(0),
                                                     surprise_strategy=surprise_strategy)
                indices_and_gains.append((sample_idx, dsa, energy[0]))
            elif coverage_al == "optimal":
                if sample_idx in selected_indices:
                    continue
                dsa, energy = self.calculate_greedy(cached_data[0][sample_idx][None, :],
                                                     cached_data[1][sample_idx].unsqueeze(0),
                                                     cached_data[2][sample_idx].unsqueeze(0),
                                                     surprise_strategy=surprise_strategy)
                if dsa > max_sample[1]:
                    max_sample = (sample_idx, float(dsa), float(energy))
        if coverage_al == "optimal":
            max_label = int(cached_data[1][max_sample[0]])
            self.SA_cache[max_label] = np.concatenate(
                (self.SA_cache[max_label], cached_data[0][max_sample[0]][None, :]), axis=0)
            return max_sample
        return indices_and_gains

    def _top_k_oneshot(self, indices_and_gains, data_loader, coverage_al, no_label=False,
                       gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        SA_batch, label_batch, score = cached_data[0], cached_data[1].to(int), cached_data[2]
        SA_batch = torch.from_numpy(SA_batch)
        indicies = list(range(label_batch.shape[0]))
        for _ in range(query_size):
            dsa2, energy2 = self.calculate_greedy_full(SA_batch, label_batch, score,
                                                        surprise_strategy=surprise_strategy)
            max_idx = np.argmax(dsa2)
            das_max = np.max(dsa2)
            en_max = np.max(energy2)
            max_sample = (indicies[max_idx], float(das_max), en_max)
            del indicies[max_idx]
            indices_and_gains.append(max_sample)
            self.SA_cache[int(label_batch[max_idx])] = np.concatenate(
                (self.SA_cache[int(label_batch[max_idx])], SA_batch[max_idx][None, :]), axis=0)
            label_batch = torch.cat((label_batch[:max_idx], label_batch[max_idx + 1:]))
            SA_batch = torch.cat((SA_batch[:max_idx], SA_batch[max_idx + 1:]))
            score = torch.cat((score[:max_idx], score[max_idx + 1:]))
        return indices_and_gains

    def _build_greedy(self, data_loader):
        cached_SA_batches = torch.tensor([])
        cached_SA_labels = torch.tensor([])
        cached_SA_scores = torch.tensor([])
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            SA_batch, label_batch, layer_output_dict, score = self.nac_forward_pass(data, label,
                                                                                     surprise_strategy="surprise",
                                                                                     use_prediction=True)
            cached_SA_batches = torch.cat((cached_SA_batches, SA_batch), dim=0)
            cached_SA_labels = torch.cat((cached_SA_labels, label_batch), dim=0)
            cached_SA_scores = torch.cat((cached_SA_scores, score), dim=0)
        return cached_SA_batches.numpy(), cached_SA_labels, cached_SA_scores

    # ── Forward Pass & Calculations ────────────────────────────────────

    def nac_forward_pass(self, data_batch, label_batch, surprise_strategy="surprise", use_prediction=False):
        """Forward pass through NAC layers, returning SA activations and scores."""
        SA_batch = []
        if self.use_nac:
            layer_output_dict, score = tool.get_layer_output_nac(self.model, data_batch,
                                                                  layer_selection=self.layer_selection, use=self.use,
                                                                  dataset=self.dataset, sigmoids=self.sigmoids)
        else:
            layer_output_dict = tool.get_layer_output(self.model, data_batch, layer_selection=self.layer_selection)

        if label_batch is not None:
            batch_size = label_batch.size(0)
        elif not self.use_nac:
            label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax().item(), device=self.device).unsqueeze(0)
            batch_size = 1
        else:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = 1
        if use_prediction:
            if not self.use_nac:
                label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax(dim=1).tolist(), device=self.device)
                if batch_size == 1:
                    label_batch = label_batch.unsqueeze(0)
            else:
                label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
                batch_size = len(label_batch)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu()  # [batch_size, num_neuron]

        return SA_batch, label_batch.detach().cpu(), layer_output_dict, score.detach().cpu()

    def calculate(self, data_batch, label_batch, surprise_strategy="coverage", use_prediction=False):
        """Calculate DSC scores for active learning (with weighted energy-DSA combination)."""
        cove_set = set()
        dsa_list = []
        energy_list = []
        dista_list = []
        distb_list = []
        SA_batch = []
        layer_output_dict, score = tool.get_layer_output_nac(self.model, data_batch,
                                                              layer_selection=self.layer_selection, use=self.use,
                                                              dataset=self.dataset, sigmoids=self.sigmoids)
        if label_batch is not None:
            batch_size = label_batch.size(0)
        else:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = 1
        if use_prediction:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = len(label_batch)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))

        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()  # [batch_size, num_neuron]
        energy = tool.get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            if self.weight_al[1] == 0 and len(dist_a_list) >= 2:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight_al[0], 1.0) * ((energy[i] - self.weight_al[1]) / self.std[1]) + \
                  max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[0])
            if surprise_strategy == "dist_a":
                dsa_list.append(float(dist_a))
            elif surprise_strategy == "dist_b":
                if dist_b > 0:
                    db = 1 / dist_b
                else:
                    db = 1e-6
                dsa_list.append(float(db))
            else:
                dsa_list.append(dsa)
                energy_list.append(energy[i])
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        if surprise_strategy == "coverage":
            return cove_set, layer_output_dict
        else:
            if batch_size == 1:
                if surprise_strategy == "dist_a":
                    return dist_a, layer_output_dict
                elif surprise_strategy == "dist_b":
                    return db, layer_output_dict
                else:
                    return dsa, energy, layer_output_dict
            else:
                return torch.tensor(dsa_list).to(self.device), torch.tensor(energy_list).to(
                    self.device), layer_output_dict

    def calculate_greedy(self, SA_batch, label_batch, score, surprise_strategy="surprise", gain_mode="standard"):
        """Calculate greedy DSC scores for a single sample (used in AL top_k)."""
        cove_set = set()
        dsa_list = []
        batch_size = label_batch.size(0)
        energy_list = []
        energy = tool.get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            if self.weight_al[1] == 0 and len(dist_a_list) >= 2:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight_al[0], 1.0) * ((energy[i] - self.weight_al[1]) / self.std[1]) + \
                  max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[0])
            if surprise_strategy == "dist_a":
                dsa_list.append(float(dist_a))
            elif surprise_strategy == "dist_b":
                if dist_b > 0:
                    db = 1 / dist_b
                else:
                    db = 1e-6
                dsa_list.append(float(db))
            else:
                dsa_list.append(dsa)
                energy_list.append(energy[i])
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        if batch_size == 1:
            if surprise_strategy == "dist_a":
                return dist_a
            elif surprise_strategy == "dist_b":
                return db
            else:
                return dsa, energy
        else:
            return torch.tensor(dsa_list).to(self.device), torch.tensor(energy_list).to(self.device)

    def calculate_greedy_full(self, SA_batch, label_batch, score, surprise_strategy="surprise"):
        """Vectorized greedy DSC calculation for all samples at once (used in optimal2 AL)."""
        energy = tool.get_energy_score(score).cpu().detach().numpy()

        lin_cache = np.concatenate([self.SA_cache[k] for k in sorted(self.SA_cache.keys())])
        idx_cache = np.concatenate([np.ones((self.SA_cache[k].shape[0])) * k for k in sorted(self.SA_cache.keys())])

        distances = torch.cdist(torch.from_numpy(lin_cache).to(self.device), SA_batch.to(self.device),
                                compute_mode='donot_use_mm_for_euclid_dist').cpu()
        cls_dist = torch.cat(
            [distances[idx_cache == k].min(dim=0)[0].unsqueeze(0) for k in sorted(self.SA_cache.keys())],
            axis=0)
        dist_a = torch.gather(cls_dist, 0, label_batch.to(torch.int64).unsqueeze(0))
        dist_b = cls_dist[(1 - torch.nn.functional.one_hot(label_batch.to(int), cls_dist.shape[0])).T.bool()].reshape(
            cls_dist.shape[0] - 1, -1)
        dist_b = dist_b.min(dim=0)[0]

        dsa = torch.where(dist_b > 0, dist_a / dist_b, 1e-6).cpu().numpy()
        dsa = min(self.weight_al[0], 1.0) * ((energy - self.weight_al[1]) / self.std[1]) + \
              max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[1])

        return dsa, energy

    # ── Coverage Tracking ──────────────────────────────────────────────

    def coverage(self, cove_set):
        return len(cove_set)

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def gain(self, cove_set_new, layer_output_dict, gain_mode="standard", surprise_strategy="coverage"):
        if surprise_strategy == "coverage":
            new_rate = self.coverage(cove_set_new)
            raw_gain = new_rate - self.current
        else:
            raw_gain = cove_set_new
        sample_gain = raw_gain
        if gain_mode == "entropy":
            p = torch.softmax(layer_output_dict["Linear-1"], -1)
            entropy = -torch.sum(p * torch.log(p + 1e-10), 1)
            sample_gain = raw_gain * entropy
        return raw_gain, sample_gain

    def _gain_base(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current, None

    def _update_base(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    # ── Save / Load ────────────────────────────────────────────────────

    def save(self, path):
        print('Saving recorded %s in %s...' % (self.name, path))
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s in %s...' % (self.name, path))
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)
