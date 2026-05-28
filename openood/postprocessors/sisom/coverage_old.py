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
        self.sigmoids = kwargs.get("sigmoids",
                                   {"AdaptiveAvgPool2d-1": 1, "Sequential-3": 1, "Sequential-2": 1, "Sequential-1": 1})
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

    def check(self, data, label, gain_mode="standard", surprise_strategy="coverage", coverage_method="NLC",
              use_prediction=False):
        total, energy, layer_output_dict, distadistb = self.calculate(data, label, surprise_strategy=surprise_strategy,
                                                                      use_prediction=use_prediction)
        return total, energy, layer_output_dict, distadistb


class SurpriseCoverage(Coverage):
    def init_variable(self, hyper, min_var=1e-5, num_class=10):
        self.name = self.get_name()
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.use = "AL"
        self.data_count = 0
        self.weight = 0
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
        if self.plot_activation_histogram:
            self.track_all_activations(self.layer_cache, self.cycle_num)
            self.layer_cache = {}
        if self.name == 'LSC':
            self.set_kde()
        if self.name == 'MDSC':
            self.to_numpy()
            res = self.estimator.class_wise_avg_cov(self.SA_cache)
            self.estimator.update(res)
            self.compute_covinv()

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

    def update_SA_history(self, SA_batch):
        # Concatenate SA_batch with SA_history along the 0th dimension
        self.SA_history = torch.cat([self.SA_history, SA_batch], dim=0)


class DSC(SurpriseCoverage):
    def get_name(self):
        return 'DSC'

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


if __name__ == '__main__':
    pass