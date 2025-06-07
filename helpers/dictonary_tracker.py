import os
import json
import pandas            as pd
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image 

from collections import defaultdict

class GenericDictonaryTracker:
    def __init__(self):
        self.items : list[dict[str, float]] = []

    def append(self, item: dict[str, float]) -> None:
        # append the dictonary items
        self.items.append(item)

    def last(self) -> dict[str, float]:
        # return the last item (will crash )
        return self.items[-1]

    def calculate_averages(self) -> dict[str, float]:
        # create default dictonary
        averages = defaultdict(list)
        for item in self.items:
            for key, value in item.items():
                averages[key].append(value)
        
        # compute the value for each item in dictonary
        for key in averages:
            averages[key] = float(sum(averages[key]) / len(averages[key]))

        # convert to normal dict before going    
        return dict(averages)

    def export_csv(self, fpath : str) -> None:
        # export to csv
        df = pd.DataFrame(self.items)
        df.to_csv(fpath, index = False)
    
    def export_json(self, fpath : str) -> None:
        # export json file
        with open(fpath, 'w+') as fout:
            json.dump(self.items, fout, indent = 2)
    

    def export_plot(self, fpath : str, title = "") -> None:
        
        # plot the data
        for key_name in self.items[0].keys():
            vals = [x[key_name] for x in self.items]
            plt.plot(vals, label = key_name)
        
        plt.xlabel("Epoch")
        plt.ylabel("Values")
        plt.title(title)
        plt.legend()
        plt.savefig(fpath)
        plt.clf()


class TrackerAndLogger:
    def __init__(self, root_dir : str, name : str, metric_to_track : str = 'ssim'):
        
        # create the directory
        self.root_dir = os.path.join(root_dir, name)
        self.root_dir = os.path.abspath(self.root_dir)
        os.makedirs(self.root_dir, exist_ok = True)

        # make subdirectories
        self.tracking_dir = self.__create_subdir("tracking")
        self.samples_dir  = self.__create_subdir("samples")
        self.weights_dir  = self.__create_subdir("weights")

        self.train_values = GenericDictonaryTracker()
        self.eval_values  = GenericDictonaryTracker()

        self.current_epoch   = 0
        self.metric_to_track = metric_to_track

        self.is_current_best  = False
        self.current_best_val = 0 

    def __create_subdir(self, subdir_name : str) -> str:
        x = os.path.join(self.root_dir, subdir_name)
        os.makedirs(x, exist_ok = True)
        return x

    def append_epoch(self, train : GenericDictonaryTracker, eval : GenericDictonaryTracker) -> None:
        
        self.is_current_best = False

        self.train_values.append(
            train.calculate_averages()
        )
        self.eval_values.append(
            eval.calculate_averages()
        )

        latest_eval = self.eval_values.last()
        value_metric = latest_eval[self.metric_to_track]
        if value_metric > self.current_best_val:
            self.is_current_best  = True
            self.current_best_val = value_metric
            print(f"| Epoch {len(self.eval_values.items)} is Best {self.current_best_val}")
        
    def write(self) -> None:
        self.train_values.export_csv(
            os.path.join(self.tracking_dir, 'train_values.csv')
        )
        self.train_values.export_plot(
            os.path.join(self.tracking_dir, 'train.png'),
            title = 'Training Values'
        )

        self.eval_values.export_csv(
            os.path.join(self.tracking_dir, 'evaluation_values.csv')
        )
        self.eval_values.export_plot(
            os.path.join(self.tracking_dir, 'evaluation.png'),
            title = 'Evaluation Values'
        )
    
    def save_samples(self, sample : torch.Tensor, fname : str) -> None:
        fpath = os.path.join(self.samples_dir, fname)
        save_image(sample, fpath)

    def current_is_best(self) -> bool:
        return self.is_current_best




    

    