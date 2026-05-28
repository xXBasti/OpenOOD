import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm
from collections import defaultdict
    
class RadiusGraphSelection:
    def __init__(self, data_points, subset_size, delta, feats, labels):

        self.labels = labels
        self.data_points = data_points
        self.subset_size = subset_size
        self.radius = delta
        self.rel_features = feats
        self.graphs = self.build_graphs()
        self.label_wise_indices = self.get_relevant_indices()

    def get_relevant_indices(self):
        unique_labels = torch.unique(self.labels)
        relevant_indices = {}
        for label in unique_labels:
            indices = (self.labels == label).nonzero().flatten()
            relevant_indices[label.item()] = indices
        return relevant_indices
    
    def build_graphs(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """

        print(f'Building a graph with radius {self.radius}')
        # distance computations are done in GPU
        cuda_feats = self.rel_features.cuda()
        unique_labels = torch.unique(self.labels)
        graphs = {}

        for label in unique_labels:
            xs, ys, ds = [], [], []
            indices = (self.labels == label).nonzero().flatten()
            label_feats = cuda_feats[indices]

            for i in tqdm(range(len(label_feats) // batch_size)):
                # distance comparisons are done in batches to reduce memory consumption
                cur_feats = label_feats[i * batch_size: (i + 1) * batch_size]
                dist = torch.cdist(cur_feats, label_feats)
                mask = dist < self.radius
                # saving edges using indices list - saves memory.
                x, y = mask.nonzero().T
                xs.append(x.cpu() + batch_size * i)
                ys.append(y.cpu())
                ds.append(dist[mask].cpu())

            xs = torch.cat(xs).numpy()
            ys = torch.cat(ys).numpy()
            ds = torch.cat(ds).numpy()

            df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
            print(f'Finished constructing graph for label {label} with radius {self.radius}')
            print(f'Graph contains {len(df)} edges.')
            graphs[label.item()] = df

        return graphs
    
    def class_balanced_graph_selection(self, num_classes):
        print(f'Start selecting {self.subset_size} samples.')
        selected = []
        class_counts = defaultdict(int)
        quota_per_class = math.ceil((self.subset_size / num_classes))
        self.lSet = np.array([])

        for label, graph_df in self.graphs.items():
            edge_from_seen = np.isin(graph_df['x'], self.lSet)
            covered_samples = set(graph_df.loc[edge_from_seen, 'y'])
            cur_df = graph_df[~graph_df['y'].isin(covered_samples)]
            degrees = np.zeros(len(self.label_wise_indices[label]), dtype=int)
            for index in cur_df['x'].values:
                degrees[index] += 1

            for i in range(self.subset_size):
                if class_counts[label] >= quota_per_class:
                    print("Quota Already Reached")
                    break  # Skip if the class quota is already met

                degrees = np.bincount(cur_df.x, minlength=len(self.label_wise_indices[label]))
                # Select the sample with the highest degree
                cur = degrees.argmax()

                # Add the selected sample to the covered set
                new_covered_samples = set(cur_df.loc[cur_df['x'] == cur, 'y'])
                assert not covered_samples.intersection(new_covered_samples), 'all samples should be new'
                cur_df = cur_df[~cur_df['y'].isin(new_covered_samples)]

                covered_samples.update(new_covered_samples)
                selected.append(int(self.label_wise_indices[label][cur]))

                # Increment the count for the selected sample's class
                class_counts[label] += 1

        subSet = np.array(self.data_points)[selected]

        print(f'Selection of {class_counts}.')
        return subSet