import os
from itertools import product
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
from sklearn.manifold import TSNE


from .base_postprocessor import BasePostprocessor

from .sisom import coverage_old as coverage

from .sisom import tool
from .sisom.coverage_al import DSC
from .sisom.kcenterGreedy import kCenterGreedyOOD
from .sisom.sampler import SubsetSequentialSampler
from .sisom.prob_cover import RadiusGraphSelection


class SisomPostprocessor(BasePostprocessor):
    """
    CoveragePostprocessor is a postprocessor class that implements coverage-based active learning for out-of-distribution (OOD) data.

    Args:
        config (dict): Configuration parameters for the postprocessor.

    Attributes:
        args (dict): Arguments extracted from the config.
        coverage_method (str): Method used for coverage calculation.
        dataset (str): Dataset name.
        coverage_hyper (float): Hyperparameter for coverage calculation.
        build_ratio (float): Ratio of samples used for building the dataloader.
        num_classes (int): Number of classes in the dataset.
        gain_mode (str): Mode for calculating gain.
        graph_radius (float): Delta value for ProbCover preselection method.
        plot_ind_ood (bool): Flag indicating whether to plot individual OOD samples.
        root_dir (str): Root directory for OOD configurations.
        sigmoids (dict): Sigmoid values for feature space optimization.
        surprise_strategy (str): Strategy for calculating surprise.
        layer_kwargs (dict): Layer keyword arguments.
        APS_mode (str): Mode for calculating APS (Average Precision Score).
        search_sigmoid (bool): Flag indicating whether to search for optimal sigmoid values.
        device (torch.device): Device used for computation.
        input_size (tuple): Input size of the network.
        build_method (str): Method used for preselection of samples.

    Methods:
        setup(net, id_loader_dict, ood_loader_dict, **kwargs):
            Sets up the postprocessor by initializing parameters and building dataloaders.
        get_features(models, data_loader, device, return_labels=False):
            Extracts features from the models using the given data loader.
        inference(net, data_loader, progress=True):
            Performs inference using the postprocessor on the given data loader.
    """
    def __init__(self, config):
        super(SisomPostprocessor, self).__init__(config)

        self.args = self.config.postprocessor.postprocessor_args
        self.coverage_method = "DSC" # self.args.coverage_method
        self.dataset = self.args.dataset
        self.num_classes = self.args.num_classes
        self.input_size = tuple(self.args.input_size)

        # Subset Selection
        self.graph_radius = self.args.graph_radius
        self.coverage_hyper = self.args.coverage_hyper
        self.build_ratio = self.args.build_ratio
        self.build_method = self.args.build_method

        # SISOM / SISOMe or Fixed Ratio:
        if self.args.fixed_ratio is not None:
            self.fixed_ratio = (True, float(self.args.fixed_ratio))
        else:
            self.fixed_ratio = (False, -1)
        self.gain_mode = self.args.gain_mode
        self.surprise_strategy = self.args.surprise_strategy # Can be removed

        # Coverage Calculation
        self.sigmoids = self.config.postprocessor.postprocessor_sweep
        self.layer_kwargs = self.config.postprocessor.layer_kwargs
        # Coverage Search mode
        self.search_sigmoid = self.config.postprocessor.search_sigmoid
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

        # self.APS_mode = self.config.postprocessor.APS_mode

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, **kwargs):
        """
        Sets up the postprocessor by initializing parameters and building dataloaders.

        Args:
            net (nn.Module): Neural network model.
            id_loader_dict (dict): Dictionary containing in-distribution (ID) dataloaders.
            ood_loader_dict (dict): Dictionary containing out-of-distribution (OOD) dataloaders.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.device=next(net.parameters()).device
        # Command Line Parameters specified in eval_ood.py
        self.graph_radius = kwargs.get("graph_radius", self.graph_radius)
        self.build_ratio = kwargs.get("build_ratio", self.build_ratio)
        self.build_method = kwargs.get("build_method", self.build_method)
        self.surprise_strategy = kwargs.get("surprise_strategy", self.surprise_strategy)
        self.fixed_ratio = kwargs.get("fixed_ratio", self.fixed_ratio)
        
        # Dataset Metadata
        random_data = torch.randn(self.input_size).to(device=self.device)
        layer_size_dict = tool.get_layer_output_sizes(net, random_data, mode="nac")
        total_samples = len(id_loader_dict["train"].dataset)
        
        # Building DataLoader with Pre-Selection Method (PC)
        # # CoreSet Preseleciton
        # if self.build_method == "coreset":
        #     feat = self.get_features(net, id_loader_dict["train"], self.device).detach().cpu().numpy()
        #     sampling = kCenterGreedyOOD(feat)
        #     _ , batch = sampling.select_batch_(torch.tensor([]), int(self.build_ratio * total_samples - 1))
        #     self.build_dataloader = DataLoader(id_loader_dict["train"].dataset,
        #                                 batch_size=128, shuffle=False,
        #                                 num_workers=8, pin_memory=True,
        #                                 sampler=SubsetSequentialSampler(batch),
        #                                 drop_last=False, collate_fn=self.collate_loader)
        #     self.stat_dataloader = DataLoader(id_loader_dict["train"].dataset,
        #                                 batch_size=128, shuffle=False,
        #                                 num_workers=8, pin_memory=True,
        #                                 drop_last=False)
        # RadiusGraph Preselection
        if self.build_method == "probcover":
            feat, labels = self.get_features(net, id_loader_dict["train"], self.device, return_labels=True)
            indices = np.arange(len(id_loader_dict["train"].dataset))
            probcov = RadiusGraphSelection(indices, subset_size=int(self.build_ratio * total_samples), delta=self.graph_radius, feats=feat, labels=labels)
            aS = probcov.class_balanced_graph_selection(self.num_classes)
            self.build_dataloader = DataLoader(id_loader_dict["train"].dataset,
                            batch_size=128, shuffle=False,
                            num_workers=8, pin_memory=True,
                            sampler=SubsetSequentialSampler(aS),
                            drop_last=False, collate_fn=self.collate_loader)
            self.stat_dataloader = DataLoader(id_loader_dict["train"].dataset,
                                        batch_size=128, shuffle=False,
                                        num_workers=8, pin_memory=True,
                                        #sampler=SubsetSequentialSampler(aS),
                                        drop_last=False)
        # Random Preselection
        elif self.build_method == "sampled":
            build_size = int(self.build_ratio * total_samples)
            stat_size = total_samples - build_size
            build_dataset, _ = random_split(id_loader_dict["train"].dataset, [build_size, stat_size])
            self.build_dataloader = DataLoader(build_dataset,
                                        batch_size=128, shuffle=False,
                                        num_workers=8, pin_memory=True,
                                        drop_last=False, collate_fn=self.collate_loader)
            self.stat_dataloader = DataLoader(build_dataset,
                                batch_size=128, shuffle=False,
                                num_workers=8, pin_memory=True,
                                drop_last=False)
        # Random Preselection
        else:
            build_size = int(total_samples)
            stat_size = total_samples - build_size
            build_dataset, _ = random_split(id_loader_dict["train"].dataset, [build_size, stat_size])
            self.build_dataloader = DataLoader(build_dataset,
                                        batch_size=128, shuffle=False,
                                        num_workers=8, pin_memory=True,
                                        drop_last=False, collate_fn=self.collate_loader)
            self.stat_dataloader = DataLoader(build_dataset,
                                batch_size=128, shuffle=False,
                                num_workers=8, pin_memory=True,
                                drop_last=False)

        # Feature Space Optimization with Sigmoids (OS)
        if self.search_sigmoid:
            self.search_optimal_sigmoids(layer_size_dict, net)

        #self.coverage_obj : coverage.Coverage = getattr(coverage, self.coverage_method)(net, layer_size_dict, hyper = self.coverage_hyper, dataset=self.dataset, sigmoids = self.layer_kwargs)
        self.coverage_obj: coverage.Coverage = DSC(net, layer_size_dict, hyper = self.coverage_hyper, dataset=self.dataset, sigmoids = self.layer_kwargs)
        self.coverage_obj.use = "OOD"
        self.coverage_obj.build(self.build_dataloader)
        self.gain_mean, self.gain_std, self.energy_mean, self.energy_std = self.calc_stats_from_train(self.stat_dataloader)
        print(f"Used Gain-Mean: {self.gain_mean.item()}, Gain-Std: {self.gain_std.item()}")
        # if gain_mode is set to energy, SISOMe is used, otherwise SISOM
        if self.gain_mode == "energy":
            self.coverage_obj.weight = self.gain_mean
        else:
            self.coverage_obj.weight = 0

    def search_optimal_sigmoids(self, layer_size_dict, net):
        combs = list(product(*self.sigmoids.values()))
        res_combs = []
        sig_combs = [{key: value for key, value in zip(self.sigmoids.keys(), comb)} for comb in combs]
        min_gain = float("inf")
        min_std = float("inf")
        print("Testing all sigmoid combinations")
        for sig_comb in tqdm(sig_combs):
            self.coverage_obj: coverage.Coverage = getattr(coverage, self.coverage_method)(net, layer_size_dict,
                                                                                           hyper=self.coverage_hyper,
                                                                                           dataset=self.dataset,
                                                                                           sigmoids=sig_comb)
            self.coverage_obj.use = "OOD"
            self.coverage_obj.build(self.build_dataloader)
            gain_mean, gain_std, _, _ = self.calc_stats_from_train(self.stat_dataloader)
            print(f"Gain-Mean: {gain_mean.item()}, Gain-Std: {gain_std.item()}, Sigmoids: {sig_comb}")
            res_combs.append((gain_mean.item(), gain_std.item(), self.dataset, self.build_ratio, sig_comb))
            if gain_mean < min_gain:
                min_gain = gain_mean
                min_std = gain_std
                self.layer_kwargs = sig_comb
        self.gain_mean, self.gain_std = min_gain, min_std
        print(
            f"Min-Gain-Mean: {self.gain_mean.item()}, Min-Gain-Std: {self.gain_std.item()}, Final-Sigmoids: {self.layer_kwargs}")
        res_combs.sort(key=lambda x: x[0])
        os.makedirs(os.path.join(self.root_dir, "sisom_search_dir"), exist_ok=True)
        pd.DataFrame(res_combs).to_csv(os.path.join(self.root_dir, "sisom_search_dir", 'weights.csv'), mode='a',
                                       header=not os.path.exists(os.path.join(self.root_dir, 'weights.csv')))

    def get_features(self, models, data_loader, device, return_labels=False):
        """
        Extracts features from the models using the given data loader.

        Args:
            models (nn.Module): Neural network models.
            data_loader (DataLoader): Data loader for loading the data.
            device (torch.device): Device used for computation.
            return_labels (bool, optional): Flag indicating whether to return the labels along with the features. Defaults to False.

        Returns:
            torch.Tensor or tuple: Extracted features or tuple of features and labels.
        """
        models.eval()
        labels = torch.tensor([])
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs in tqdm(data_loader):
                data = inputs["data"]
                label = inputs["label"]
                with torch.cuda.device(device):
                    data = data.cuda()
                    _, features_batch = models(data, return_feature=True)
                if return_labels:
                    labels = torch.cat((labels, label.cpu()), 0)
                features = torch.cat((features, features_batch), 0)
            feat = features  # .detach().cpu().numpy()
        if return_labels:
            return feat.detach().cpu(), labels
        return feat

    def inference(self, net, data_loader, progress=True):

        """
        Performs inference using the postprocessor on the given data loader.

        Args:
            net (nn.Module): Neural network model.
            data_loader (DataLoader): Data loader for loading the data.
            progress (bool, optional): Flag indicating whether to display progress bar. Defaults to True.

        Returns:
            tuple: Tuple containing predicted labels, confidence scores, and true labels.
        """
        pred_list, conf_list, label_list = [], [], []
        test_ds = data_loader.dataset
        data_loader = DataLoader(test_ds,
                        batch_size=128, shuffle=False,
                        num_workers=8, pin_memory=True,
                        drop_last=False)
        SA_batches = torch.tensor([])
        statistics = torch.tensor([])
        print(self.fixed_ratio,self.gain_mean)
        for b in tqdm(data_loader):
            SA_batch = []
            data = b["data"]
            label = b["label"]
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            # Get distance ratio r (gain) and energy. The other return values are for plotting
            gain, energy, layer_output_dict, (dista, distb) = self.coverage_obj.check(data, label, surprise_strategy=self.surprise_strategy, 
                                                  coverage_method = self.coverage_method, use_prediction = True)
            # Weighted Combination of Gain and Energy
            if self.surprise_strategy == "energy":
                if self.fixed_ratio[0]:
                    self.gain_mean = torch.tensor(self.fixed_ratio[1])

                total_gain = min(self.gain_mean, 1.0) * ((energy-self.energy_mean)/self.energy_std) \
                    + max(1.0 - self.gain_mean, 0.0) * ((gain-self.gain_mean)/self.gain_std)
            else:
                total_gain = (gain-self.gain_mean)/self.gain_std
            # print(total_gain)
            # total_gain = torch.sigmoid(total_gain)
            # total_gain = 1-((total_gain+1.0)/2.0)
            total_gain = - total_gain
            # print(total_gain)

            pred_list.append(layer_output_dict["Linear-1"].argmax(dim=1).cpu())
            if data_loader.batch_size == 1:
                total_gain = total_gain.unsqueeze(0)
            conf_list.append(total_gain.cpu())
            label_list.append(label.cpu())

        # Return Values
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
        
    def calc_stats_from_train(self, data_loader: DataLoader):
            """
            Calculates the mean gain and energy from the training set.

            Args:
                data_loader (DataLoader): The data loader containing the training data.

            Returns:
                Tuple: A tuple containing the mean gain, gain standard deviation, mean energy, and energy standard deviation.
            """
            gain_list = torch.tensor([])
            energy_list = torch.tensor([])
            for b in tqdm(data_loader):
                data = b["data"]
                label = b["label"]
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                label = label.to(self.device)
                single_gain, single_energy, _, _= self.coverage_obj.check(data, label, surprise_strategy=self.surprise_strategy, 
                                                      coverage_method = self.coverage_method)
                single_gain = single_gain.cpu()
                single_energy = single_energy.cpu()
                if data_loader.batch_size == 1:
                    single_gain = single_gain.unsqueeze(0)
                gain_list = torch.cat((gain_list.cpu(), single_gain), dim=0)
                energy_list = torch.cat((energy_list.cpu(), single_energy), dim=0)
            
            gain_mean = torch.mean(gain_list)
            gain_std = torch.std(gain_list)
            energy_mean = torch.mean(energy_list)
            energy_std = torch.std(energy_list)
            return gain_mean, gain_std, energy_mean, energy_std
    
    def collate_loader(self, batch):
        """Selects 'data' and 'label' properties from sample dictionaries"""
        data = [sample['data'] for sample in batch]
        labels = [sample['label'] for sample in batch]
        data = torch.stack(data, dim=0)
        labels = torch.tensor(labels)
        return data, labels
