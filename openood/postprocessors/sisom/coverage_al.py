from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from . import tool


class Coverage:
    def __init__(self, model, layer_size_dict, cycle_num=0, hyper=None, dataset="cifar10", **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset
        self.layer_size_dict = layer_size_dict
        self.cycle_num = cycle_num
        self.sigmoids = kwargs.get("sigmoids", None)
        kwargs.pop('sigmoids', None)
        self.init_variable(hyper, **kwargs)

    def init_variable(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        with torch.no_grad():
            for data, *_ in tqdm(data_loader):
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                self.step(data)

    def assess_ood(self, data_loader, coverage_method="NC", surprise_strategy="coverage"):
        gain_list = torch.tensor([])
        with torch.no_grad():
            for b in tqdm(data_loader):
                data = b["data"]
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                if "SC" not in coverage_method:
                    raw_gain = self.step(data).unsqueeze(0).cpu()
                else:
                    label = b["label"]
                    raw_gain, _ = self.step(data, label, surprise_strategy=surprise_strategy)
                gain_list = torch.cat((gain_list.cpu(), raw_gain))
        return gain_list

    def step(self, data, label=None, *args, **kwargs):
        cove_dict, layer_output_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain is not None:
            self.update(cove_dict, gain)
            if isinstance(gain, tuple):
                return gain[0]
            else:
                return gain
        # TODO: for nlc, two options: 1. check if nlc and then update and return total (gain[0]) if gain is not None, else return 0
        # 2. option: always update, return total[0] if nlc and gain not none, else 0
        else:
            return torch.tensor(0)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    def check(self, data, label, gain_mode="standard", surprise_strategy="coverage", coverage_method="NLC",
              use_prediction=False):

        total, energy, layer_output_dict, distadistb = self.calculate_ood(data, label,
                                                                          surprise_strategy=surprise_strategy,
                                                                          use_prediction=use_prediction)
        return total, energy, layer_output_dict, distadistb

    def top_k_loop(self, indices_and_gains: list, data_loader, coverage_al: str, no_label: bool = False,
                   gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        # max_sample keeps the top sample by saving gain, index and data
        max_sample = (-1, -1.0, -1)
        # Loop through the data loader
        selected_indices = [idx for idx, _, _ in indices_and_gains]
        dataset_size = len(data_loader.dataset)
        for i in tqdm(range(cached_data[0].shape[0])):
            # Get the index for the current sample
            sample_idx = i
            if coverage_al == "incremental":
                dsa, energy = self.calculate_greedy(cached_data[0][sample_idx][None, :],
                                                    cached_data[1][sample_idx].unsqueeze(0),
                                                    cached_data[2][sample_idx].unsqueeze(0),
                                                    surprise_strategy=surprise_strategy)
                # dsa, energy = self.step(sample_data, sample_target, gain_mode = gain_mode, surprise_strategy = surprise_strategy)
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

    def top_k_oneshot(self, indices_and_gains: list, data_loader, coverage_al: str, no_label: bool = False,
                      gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        """
        :param indices_and_gains:
        :param data_loader:
        :param coverage_al:
        :param no_label:
        :param gain_mode:
        :param surprise_strategy:
        :param query_size:
        :param cached_data:

        :return indices_and_gains
        """
        SA_batch, label_batch, score = cached_data[0], cached_data[1].to(int), cached_data[2]
        SA_batch = torch.from_numpy(SA_batch)
        indicies = list(range(label_batch.shape[0]))
        for _ in range(query_size):
            dsa2, energy2 = self.calculate_greedy_full(SA_batch,
                                                       label_batch,
                                                       score,
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

    def top_k(self, data_loader, k, coverage_al, no_label=False, gain_mode="standard", surprise_strategy="coverage"):
        # Initialize a min-heap to keep track of the top k indices with highest gains
        indices_and_gains = []
        # top_indices_heap = []
        cached_SA_batches, cached_SA_labels, cached_SA_scores = self.build_greedy(data_loader)
        cached_data = (cached_SA_batches, cached_SA_labels, cached_SA_scores)
        if coverage_al == "optimal":
            for _ in tqdm(range(k)):
                top_sample = self.top_k_loop(indices_and_gains, data_loader, coverage_al, query_size=k,
                                             cached_data=cached_data)
                # self.SA_cache[top_sample.label].append(top_sample.data)
                indices_and_gains.append(top_sample)
        elif coverage_al == "optimal2":
            indices_and_gains = self.top_k_oneshot(indices_and_gains, data_loader, coverage_al, query_size=k,
                                                   cached_data=cached_data)
            # self.SA_cache[top_sample.label].append(top_sample.data)
        elif coverage_al == "individual" or coverage_al == "incremental":
            indices_and_gains = self.top_k_loop(indices_and_gains, data_loader, coverage_al, no_label=no_label,
                                                gain_mode=gain_mode, surprise_strategy=surprise_strategy, query_size=k,
                                                cached_data=cached_data)

        # Sort the top k indices by their order of appearance in the data loader
        # top_indices_and_gains = sorted([(idx, gain) for gain, idx in top_indices_heap])

        indices_and_gains.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _, _ in indices_and_gains]
        top_gains = [gain.cpu().item() if torch.is_tensor(gain) else gain for _, gain, _ in indices_and_gains]
        top_energies = [energy.cpu().item() if torch.is_tensor(energy) else energy for _, _, energy in
                        indices_and_gains]
        return top_gains, top_indices

    def build_greedy(self, data_loader):
        # cached SA_batches contains all SA_batches, has size (num_samples, num_neurons)
        # initialize SA_batches as empty numpy array
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
            # SA_batch is a numpy array of shape (batch_size, num_neurons)
            cached_SA_batches = torch.cat((cached_SA_batches, SA_batch), dim=0)
            cached_SA_labels = torch.cat((cached_SA_labels, label_batch), dim=0)
            cached_SA_scores = torch.cat((cached_SA_scores, score), dim=0)
        return cached_SA_batches.numpy(), cached_SA_labels, cached_SA_scores

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current, None

    def calculate_greedy(self, *arg, **kwargs):
        raise NotImplementedError("Must be overwritten in child classes")


class SurpriseCoverage(Coverage):
    def init_variable(self, hyper, min_var=1e-5, num_class=10):
        self.name = self.get_name()
        assert self.name in ['LSC', 'DSC', 'MDSC']
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
        self.SA_history = torch.tensor([], device=self.device)
        self.layer_selection = None
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def get_name(self):
        raise NotImplementedError

    def build(self, data_loader):
        # with torch.no_grad():
        print('Building Mean & Var...')
        if self.min_var > 0:
            for i, (data, label) in enumerate(tqdm(data_loader)):
                # print(data.size())
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                self.set_meam_var(data, label)
        self.set_mask()
        # with torch.no_grad():
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, label)
        self.to_numpy()

    def assess(self, data_loader, gain_mode="standard", surprise_strategy="surprise"):
        if ("energy" in gain_mode):
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
        dsa, energy, layer_output_dict = self.calculate(data, label, surprise_strategy)
        # raw_gain, sample_gain = self.gain(cove_set, layer_output_dict, gain_mode, surprise_strategy)
        return dsa, energy

    def set_meam_var(self, data, label):
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

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, label_batch):
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
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new, layer_output_dict, gain_mode="standard", surprise_strategy="coverage"):
        if surprise_strategy == "coverage":
            new_rate = self.coverage(cove_set_new)
            raw_gain = new_rate - self.current
        else:
            raw_gain = cove_set_new
        if gain_mode == "standard" or ("energy" in gain_mode):
            sample_gain = raw_gain
        elif "similarity" in gain_mode:
            # implement sim gain
            SA_batch = []
            for (layer_name, layer_output) in layer_output_dict.items():
                if "Linear" in layer_name and self.use_nac:
                    continue
                else:
                    SA_batch.append(
                        layer_output[:, self.mask_index_dict[layer_name]].view(layer_output.shape[0], -1).to(
                            device=self.device))
            SA_batch = torch.cat(SA_batch, 1).detach().to(device=self.device)  # [batch_size, num_neuron]
            # Compute the average cosine similarity for each sample in SA_batch
            if "cover" not in gain_mode:
                average_similarity_batch = self.compute_similarity(SA_batch, gain_mode)
                sample_gain = (raw_gain * (1 - average_similarity_batch)).cpu()
            else:
                sample_gain = raw_gain
            # Update SA_history by adding samples from SA_batch
            self.update_SA_history(SA_batch)
        elif gain_mode == "entropy":
            p = torch.softmax(layer_output["Linear-1"], -1)
            entropy = -torch.sum(p * torch.log(p + 1e-10), 1)
            sample_gain = raw_gain * entropy

        return raw_gain, sample_gain

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

    def compute_similarity(self, SA_batch, similarity_method="avg_similarity", beta=0.95):
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(SA_batch)

        # Calculate the desired similarity based on the chosen method
        if similarity_method == "avg_similarity" or similarity_method == "similarity":
            result = torch.mean(similarity, dim=1)
        elif similarity_method == "max_similarity":
            result = torch.max(similarity, dim=1).values
        elif similarity_method == "exp_similarity":
            result = self.compute_exp_weighted_average_similarity(similarity, beta)
        else:
            raise ValueError("Invalid similarity method.")

        return result

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def compute_cosine_similarity(self, SA_batch, SA_bank=None):
        # Reshape SA_batch to have dimensions (batch_size, 1, num_neurons)
        SA_batch = SA_batch.to(device=self.device)
        if SA_bank == None:
            SA_bank = self.SA_history
        if SA_bank.numel() == 0:  # Check if SA_history is empty
            return torch.zeros(SA_batch.size(0), SA_batch.size(0),
                               device=self.device)  # Return zeros for similarity if SA_history is empty

        # Compute cosine similarity using PyTorch's cosine_similarity function
        similarity = self.sim_matrix(SA_batch, SA_bank.to(self.device))

        return similarity.cpu()

    def compute_exp_weighted_average_similarity(self, similarity, beta):
        # Calculate the exponentially weighted average similarity for each sample in SA_batch
        num_samples = self.SA_history.size(0)
        exp_weights = torch.tensor([beta ** (num_samples - i - 1) for i in range(num_samples)],
                                   device=self.SA_history.device)
        exp_weights = exp_weights / exp_weights.sum()

        exp_weighted_avg_similarity = torch.matmul(similarity, exp_weights)

        return exp_weighted_avg_similarity

    def update_SA_history(self, SA_batch):
        # Concatenate SA_batch with SA_history along the 0th dimension
        self.SA_history = torch.cat([self.SA_history, SA_batch], dim=0)


class DSC(SurpriseCoverage):
    def get_name(self):
        return 'DSC'

    def nac_forward_pass(self, data_batch, label_batch, surprise_strategy="surprise", use_prediction=False):
        SA_batch = []
        if self.use_nac:
            layer_output_dict, score = tool.get_layer_output_nac(self.model, data_batch,
                                                                 layer_selection=self.layer_selection, use=self.use,
                                                                 dataset=self.dataset, sigmoids=self.sigmoids)
        else:
            layer_output_dict = tool.get_layer_output(self.model, data_batch, layer_selection=self.layer_selection)

        # it has to be considered that the actual label is supposed to be unknown during inference
        # thus, one should use the predicted label for computing the SA-Coverage instead
        # TODO: generalize later, for now, assume classification layer is called "linear-1"
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

    def calculate_ood(self, data_batch, label_batch, surprise_strategy="coverage", use_prediction=False):
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
            if self.weight == 0:
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

    def calculate(self, data_batch, label_batch, surprise_strategy="coverage", use_prediction=False):
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

            # # using numpy
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            if self.weight_al[1] == 0:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight_al[0], 1.0) * ((energy[i] - self.weight_al[1]) / self.std[1]) + \
                  max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[0])
            # dsa = energy[i]
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
        cove_set = set()
        dsa_list = []
        batch_size = label_batch.size(0)
        energy_list = []
        energy = tool.get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            # # using numpy
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)
            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            if self.weight_al[1] == 0:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight_al[0], 1.0) * ((energy[i] - self.weight_al[1]) / self.std[1]) + \
                  max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[0])
            # dsa = energy[i]
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
        energy = tool.get_energy_score(score).cpu().detach().numpy()

        # lin cache [samples x dim]
        lin_cache = np.concatenate([self.SA_cache[k] for k in sorted(self.SA_cache.keys())])
        idx_cache = np.concatenate([np.ones((self.SA_cache[k].shape[0])) * k for k in sorted(self.SA_cache.keys())])

        distances = torch.cdist(torch.from_numpy(lin_cache).to(self.device), SA_batch.to(self.device),
                                compute_mode='donot_use_mm_for_euclid_dist').cpu()  # [cache_smaples x new samples]
        cls_dist = torch.cat(
            [distances[idx_cache == k].min(dim=0)[0].unsqueeze(0) for k in sorted(self.SA_cache.keys())],
            axis=0)  # [cls x new samples]
        dist_a = torch.gather(cls_dist, 0, label_batch.to(torch.int64).unsqueeze(0))  # [ 1 x new samples]
        dist_b = cls_dist[(1 - torch.nn.functional.one_hot(label_batch.to(int), cls_dist.shape[0])).T.bool()].reshape(
            cls_dist.shape[0] - 1, -1)  # [(cls -1) x new samples]
        dist_b = dist_b.min(dim=0)[0]  # [ 1 x new samples]

        dsa = torch.where(dist_b > 0, dist_a / dist_b, 1e-6).cpu().numpy()
        dsa = min(self.weight_al[0], 1.0) * ((energy - self.weight_al[1]) / self.std[1]) + \
              max(1.0 - self.weight_al[0], 0.0) * ((dsa - self.weight_al[0]) / self.std[1])

        return dsa, energy


if __name__ == '__main__':
    pass