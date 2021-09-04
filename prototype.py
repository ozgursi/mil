import pandas as pd
import sys
import numpy as np
from sklearn import tree, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import math
import itertools

import os
import random

import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import math
import itertools

import os

import numpy as np
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from model import ShapeletGenerator, pairwise_dist
from mil import get_data
#from prototype_forest import PrototypeForest
import time
import os

import torch
import torch.nn as nn
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def gram_matrix(mat):
  mat = mat.squeeze(dim=0)
  mat = torch.mm(mat, mat.t())
  return mat



def pairwise_dist(x, y):
  x_norm = (x.norm(dim=2)[:, :, None])
  y_t = y.permute(0, 2, 1).contiguous()
  y_norm = (y.norm(dim=2)[:, None])
  y_t = torch.cat([y_t] * x.shape[0], dim=0)
  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
  return torch.clamp(dist, 0.0, np.inf)

class ShapeletGenerator(nn.Module):

    def __init__(self, n_prototypes, bag_size, n_classes, features):
        n_prototypes = int(n_prototypes)
        super(ShapeletGenerator, self).__init__()

        number_of_rows = features.shape[0]

        random_indices = np.random.choice(number_of_rows, 
                                          size=1, 
                                         replace=False)
        
        prot = features[random_indices, :]
        prot = prot.reshape(1, n_prototypes, prot.shape[1])
        prot = prot.astype("float32")
        self.prototypes = torch.from_numpy(prot).requires_grad_()
        #self.prototypes = (torch.randn(
        #    (1, n_prototypes, bag_size))).requires_grad_()
        if n_classes == 2:
            n_classes = 1
        self.linear_layer = torch.nn.Linear(3 * n_prototypes, n_classes, bias=False)
        #self.linear_layer.weight = torch.nn.Parameter(self.linear_layer.weight/100000)
        self.n_classes = n_classes

    def pairwise_distances(self, x, y):
        x_norm = (x.norm(dim=2)[:, :, None])
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = (y.norm(dim=2)[:, None])
        y_t = torch.cat([y_t] * x.shape[0], dim=0)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def get_output(self, batch_inp):
        dist = self.pairwise_distances(batch_inp, self.prototypes)
        min_dist = dist.min(dim=1)[0]
        max_dist = dist.max(dim=1)[0]
        mean_dist = dist.mean(dim=1)
        all_features = torch.cat([min_dist, max_dist, mean_dist], dim=1)
        logits = self.linear_layer(all_features)

        return logits, all_features

    def forward(self, x):
        logits, distances = self.get_output(x)
        if self.n_classes == 1:
          logits = logits.view(1)
        return logits, distances


def convert_to_bags(data,
                    split_instances=False,
                    instance_norm=True,
                    split_ratio=0.2,
                    stride_ratio=0.5):
  bags = []
  labels = []
  current_bag = []
  current_label = data[0, 0]
  cur = data[0, 1]
  instance_size = np.round(split_ratio * data[0, 2:].shape[0]).astype("int")
  stride = np.round(stride_ratio * instance_size).astype("int")

  for i in range(data.shape[0]):
    if data[i, 1] == cur:
      instance = data[i, 2:]
      if instance_norm:
        instance = (instance - np.mean(instance)) / (1e-08 + np.std(instance))
      if split_instances:
        size = instance.shape[0]
        window = instance_size
        while True:
          current_bag.append(instance[window - instance_size:window])
          window += stride
          if window >= size:
            window = size
            current_bag.append(instance[window - instance_size:window])
            break
      else:
        current_bag.append(instance)
    else:
      bags.append(np.array(current_bag))
      labels.append(np.array(current_label))
      current_label = data[i, 0]
      current_bag = []
      instance = data[i, 2:]
      if instance_norm:
        instance = (instance - np.mean(instance)) / (1e-08 + np.std(instance))
      if split_instances:
        size = instance.shape[0]
        window = instance_size
        while True:
          current_bag.append(instance[window - instance_size:window])
          window += stride
          if window >= size:
            window = size
            current_bag.append(instance[window - instance_size:window])
            break
      else:
        current_bag.append(instance)
      cur = data[i, 1]
  bags.append(np.array(current_bag))
  labels.append(np.array(current_label, dtype="int32"))
  return bags, labels

def find_prototype(bags,
                   features,
                   labels,
                   early_stopping_round = 10):
    
    n_classes=2
    n_epochs=100
    batch_size=1
    display_every=5
    final_vals = []
    reg_lambda_dist = generate_random(parameters[0][0], parameters[0][1])
    reg_lambda_w = generate_random(parameters[1][0], parameters[1][1])
    reg_lambda_p = generate_random(parameters[2][0], parameters[2][1])
    lr_prot = generate_random(parameters[3][0], parameters[3][1])
    lr_weights = generate_random(parameters[4][0], parameters[4][1])
    reg_w = 1
    n_prototypes = 1
    #reg_lambda_dist = 0.0005
    #reg_lambda_w = 0.005
    #reg_lambda_p = 0.00005
    #lr_prot = 0.00001
    #lr_weights = 0.00001
    #reg_w = 1
    #n_prototypes = 2
    #n_prototypes = n_prototypes*2
    
    data1 = np.vstack((labels, bags)).T
    data = np.concatenate([data1, features], axis=1)
    
    bags_train, labels_train = convert_to_bags(data)
    bags_train = np.array(bags_train)
    labels_train = np.array(labels_train)

    for rep in range(1, 2):
        vals = []
        for fold in range(1, 2):
            accs = [] 

            use_cuda = False

            bag_size = bags_train[0][0].shape[0]
            #step_per_epoch = len(bags_train)
            step_per_epoch = len(np.unique(bags))

            lr_step = (step_per_epoch * 40)
            display = (step_per_epoch * display_every)
            max_steps = n_epochs * step_per_epoch
            
            model = ShapeletGenerator(n_prototypes, bag_size, n_classes, features)

            if n_classes == 2:
                output_fn = torch.nn.Sigmoid()
            else:
                output_fn = torch.nn.Softmax()



            if n_classes == 2:
                loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
            else:
                loss = torch.nn.CrossEntropyLoss(reduction="mean")

            optim1 = torch.optim.Adam([model.prototypes], lr=lr_prot)
            optim2 = torch.optim.Adam(list(model.linear_layer.parameters()),
                        lr=lr_weights)
            total_loss = 0
            correct = 0
            train_loss_hist, eval_loss_hist = [], []
            train_acc_hist, eval_acc_hist = [], []
            eval_aucs = []
            step_hist = []
            time_hist = []

            if use_cuda and torch.cuda.is_available():
                model = model.cuda()

            cont = True
            
            max_stagnation = 0 # number of epochs without improvement to tolerate
            best_prototype = None
            best_score = 0
            i = 0
            
            while i < max_steps and max_stagnation < early_stopping_round:
                i += 1
                np_idx = np.random.choice(bags_train.shape[0], batch_size)
                start_time = time.time()
                batch_inp = bags_train[np_idx]
                targets = torch.Tensor(labels_train[np_idx]).type(torch.int64)
                batch_inp = torch.Tensor(batch_inp[0])
                batch_inp = batch_inp.view(1, batch_inp.shape[0], batch_inp.shape[1])
                if use_cuda and torch.cuda.is_available():
                    targets = targets.cuda()
                    batch_inp = batch_inp.cuda()

                logits, distances = model(batch_inp)
                out = output_fn(logits)

                if n_classes == 2:
                    predicted = (out > 0.5).type(torch.int64)
                else:
                    _, predicted = torch.max(out, 1)
                correct += (predicted == targets).type(torch.float32).mean().item()

                batch_loss = loss(logits, targets.type(torch.float32))

                prototypes_pairwise = pairwise_dist(model.prototypes, model.prototypes)
                reg_prototypes = prototypes_pairwise.sum()

                weight_reg = 0
                for param in model.linear_layer.parameters():
                    weight_reg += param.norm(p=reg_w).sum()

                reg_loss = reg_lambda_w*weight_reg + reg_lambda_dist*distances.sum() - reg_prototypes*reg_lambda_p
                total_loss += batch_loss
                min_loss = batch_loss + reg_loss
                min_loss.backward()

                optim1.step()
                optim2.step()

                if (i + 1) % lr_step == 0:
                    print("LR DROP!")
                    optims = [optim1, optim2]
                    for o in optims:
                        for p in o.param_groups:
                            p["lr"] = p["lr"] / 2

                if (i + 1) % display == 0:
                    with torch.no_grad():
                        #print("Step : ", str(i + 1), "Loss: ",
                        #total_loss.item() / display, " accuracy: ", correct / (display))
                        train_loss_hist.append(total_loss.item() / display)
                        train_acc_hist.append(correct / display)
                        total_loss = 0
                        correct = 0
                        model = model.eval()
                        e_loss = 0
                        e_acc = 0
                        y_true = []
                        y_score = []

                        for i in range(len(bags_train)):
                            batch_inp = torch.Tensor(bags_train[i])
                            batch_inp = batch_inp.view(1, batch_inp.shape[0],
                                                  batch_inp.shape[1])
                            targets = torch.Tensor([labels_train[i]]).type(torch.int64)
                            logits, distances = model(batch_inp)
                            out = output_fn(logits)

                            if n_classes == 2:
                                predicted = (out > 0.5).type(torch.int64)
                            else:
                                _, predicted = torch.max(out, 1)
                            y_true.append(targets)
                            y_score.append(out)
                            correct = (predicted == targets).type(torch.float32).mean().item()
                            e_acc += correct
                            eval_loss = loss(logits, targets.type(torch.float32)).item()
                            e_loss += eval_loss

                        y_true_list = [x.tolist() for x in y_true]
                        y_score_list = [x.tolist() for x in y_score]
                        score_auc = roc_auc_score(y_true_list, y_score_list)
                        #print("Eval Loss: ", e_loss / len(bags_train),
                        #    " Eval Accuracy:", e_acc / len(bags_train), " AUC: ",
                        #score_auc)
                        
                        if score_auc > best_score:
                            best_score = score_auc
                            best_prototype = model.prototypes
                            max_stagnation = 0
                        else:
                            max_stagnation += 1
                        
                        #print("max_stagnation ", max_stagnation)
                        eval_loss_hist.append(e_loss / len(bags_train))
                        eval_acc_hist.append(e_acc / len(bags_train))
                        eval_aucs.append(roc_auc_score(y_true_list, y_score_list))
                        accs.append(e_acc / len(bags_train))
                        step_hist.append(i+1)
                        model = model.train()

    return best_prototype

class Node:
    def __init__(self):

        self.right = None
        self.left = None
        
        self.prototype = None
        
        self.column = None
        self.threshold = None
        
        self.probas = None
        self.depth = None
        
        self.is_terminal = False
        
class PrototypeTreeClassifier:
    def __init__(self,
                train_features,
                 feature_types = ["min", "max", "mean"], 
                 max_depth = 3, 
                 min_samples_leaf = 1, 
                 min_samples_split = 2, 
                 prototype_count = 1,
                 use_prototype_learner=True,
                 early_stopping_round = 3):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.prototype_count = prototype_count
        self.feature_types = feature_types
        self.train_features = train_features
        self.use_prototype_learner = use_prototype_learner
        self.Tree = None
        self.early_stopping_round = early_stopping_round
        
    def prototype(self, bags, features, labels, prototype_count):
        if self.use_prototype_learner:
            prototypes = find_prototype(bags, features, labels, self.early_stopping_round)
            check = prototypes.cpu().detach().numpy()

            check.resize(check.shape[1], check.shape[2])
            
            return check
        
        else:
            number_of_rows = self.train_features.shape[0]
            random_indices = np.random.choice(number_of_rows, 
                                              size=prototype_count, 
                                              replace=False)
            
            prot = self.train_features[random_indices, :]
            if len(prot.shape) == 1:
                prot = prot.reshape(1, prot.shape[0])
            return prot

    def nodeProbas(self, y):
        # for each unique label calculate the probability for it
        probas = []

        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def features_via_prototype(self, feature_types, features, bag_ids, prototypes):
        distances = self.calculate_distances(features, prototypes)

        bin_count  = np.unique(bag_ids, return_counts=True)[1]
        ids, index  = np.unique(bag_ids, return_index=True)

        feature_list = []
        for i in range(0, prototypes.shape[0]):
            if "max" in feature_types:
                group_max = np.maximum.reduceat(distances[:, i], index)
                max_vals = np.repeat(group_max, bin_count)
                feature_list.append(max_vals)

            if "min" in feature_types:
                group_min = np.minimum.reduceat(distances[:, i], index)
                min_vals = np.repeat(group_min, bin_count)
                feature_list.append(min_vals)

            if "mean" in feature_types:
                group_sum = np.add.reduceat(distances[:, i], index)
                group_mean = np.add.reduceat(distances[:, i], index)
                mean_vals = np.repeat(group_mean, bin_count)
                feature_list.append(mean_vals)

        return np.array(np.transpose(feature_list))

    def dist1d(self, features, prototypes, distance_type="l2"):
        if distance_type == "l2":
            distance = np.linalg.norm(features - prototypes, axis=1)
        elif distance_type == "l1":
            distance = np.abs(features - prototypes)
            distance = np.sum(distance, axis=1)

        return distance

    def calculate_distances(self, features, prototypes):
        feature_list = []
        for i in range(0, prototypes.shape[0]):
            data = self.dist1d(features, prototypes[i], distance_type="l2")
            feature_list.append(data)
        data = np.column_stack(feature_list)

        return data

    def calcBestSplit(self, features, features_via_prototype, labels, bag_ids):
        bdc = tree.DecisionTreeClassifier(
            random_state=0, 
            max_depth=1, 
            criterion="entropy",
            min_samples_split=2
        )
        bdc.fit(features_via_prototype, labels.flatten())

        threshold = bdc.tree_.threshold[0]
        split_col = bdc.tree_.feature[0]

        features_left = features[features_via_prototype[:,split_col] <= bdc.tree_.threshold[0]]
        features_right = features[features_via_prototype[:,split_col] > bdc.tree_.threshold[0]]

        labels_left = labels[features_via_prototype[:,split_col] <= bdc.tree_.threshold[0]]
        labels_right = labels[features_via_prototype[:,split_col] > bdc.tree_.threshold[0]]

        bag_ids_left = bag_ids[features_via_prototype[:,split_col] <= bdc.tree_.threshold[0]]
        bag_ids_right = bag_ids[features_via_prototype[:,split_col] > bdc.tree_.threshold[0]]

        return split_col, threshold, features_left, features_right, labels_left, labels_right, bag_ids_left, bag_ids_right

    def buildDT(self, features, labels, bag_ids, node):
            '''
            Recursively builds decision tree from the top to bottom
            '''
            # checking for the terminal conditions

            if node.depth >= self.max_depth:
                node.is_terminal = True
                return

            if features.shape[0] < self.min_samples_split:
                node.is_terminal = True
                return

            if np.unique(labels).shape[0] == 1:
                node.is_terminal = True
                return

            node.prototype = self.prototype(bag_ids, features, labels, self.prototype_count)
            features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, node.prototype)
            # calculating current split
            (splitCol, 
             thresh, 
             features_left, 
             features_right, 
             labels_left, 
             labels_right, 
             bag_ids_left, 
             bag_ids_right) = self.calcBestSplit(features, 
                                                 features_updated, 
                                                 labels, 
                                                 bag_ids)

            if splitCol is None:
                node.is_terminal = True
                return

            if features_left.shape[0] < self.min_samples_leaf or features_right.shape[0] < self.min_samples_leaf:
                node.is_terminal = True
                return

            node.column = splitCol
            node.threshold = thresh

            # creating left and right child nodes
            node.left = Node()
            node.left.depth = node.depth + 1
            node.left.probas = self.nodeProbas(labels_left)

            node.right = Node()
            node.right.depth = node.depth + 1
            node.right.probas = self.nodeProbas(labels_right)

            # splitting recursevely

            self.buildDT(features_right, labels_right, bag_ids_right, node.right)
            self.buildDT(features_left, labels_left, bag_ids_left, node.left)

    def fit(self, features, labels, bag_ids):
        '''
        Standard fit function to run all the model training
        '''
        self.classes = np.unique(labels)

        self.Tree = Node()
        self.Tree.depth = 1

        self.buildDT(features, labels, bag_ids, self.Tree)

    def predictSample(self, features, bag_ids, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas

        features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, node.prototype)

        if features_updated[0][node.column] > node.threshold:
            probas = self.predictSample(features, bag_ids, node.right)
        else:
            probas = self.predictSample(features, bag_ids, node.left)

        return probas

    def predict(self, features, bag_ids):
        '''
        Returns the labels for each X
        '''

        if type(features) == pd.DataFrame:
            X = np.asarray(features)

        sort_index = np.argsort(bag_ids)
        bag_ids = bag_ids[sort_index]
        features = features[sort_index]

        features_updated = self.features_via_prototype(self.feature_types, features, bag_ids, self.Tree.prototype)

        index  = np.unique(bag_ids, return_index=True)[1]
        count  = np.unique(bag_ids, return_counts=True)[1]
        index = np.append(index, bag_ids.shape[0])   
        predictions = []

        for i in range(0, len(index) - 1):
            pred = np.argmax(self.predictSample(features[index[i]:index[i+1]], 
                                                bag_ids[index[i]:index[i+1]], 
                                                self.Tree))
            pred = np.repeat(pred, count[i])
            predictions = np.concatenate((predictions, pred), axis=0)

        return np.asarray(predictions)
    
def sample(features, labels, bag_ids, stratified, sample_rate):
    if stratified:
        pos_sample_size = math.ceil(np.where(labels == 1)[0].shape[0] * sample_rate)
        neg_sample_size = math.ceil(np.where(labels == 0)[0].shape[0] * sample_rate)
        indices_pos = np.random.choice(np.where(labels == 1)[0], pos_sample_size, replace=False)
        indices_neg = np.random.choice(np.where(labels == 0)[0], neg_sample_size, replace=False)
        inbag_indices = np.concatenate((indices_pos, indices_neg))
    else:
        sample_size = math.ceil(labels.shape[0] * sample_rate)
        inbag_indices = np.random.choice(np.where(labels == 1)[0], sample_size, replace=False)

    oo_bag_mask = np.ones(labels.shape[0], dtype=bool)
    oo_bag_mask[inbag_indices] = False

    outbag_indices = np.where(oo_bag_mask == 1)

    return inbag_indices, outbag_indices

def get_parameter_scores(features, labels, bag_ids, params, fit_on_full = True):
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    param_vals_scores = dict()
    for param_vals in params_list:
        if param_vals["explained_variance"] < 1:
            pipe = Pipeline([('pca', PCA(n_components = param_vals["explained_variance"], 
                             svd_solver = "full")), 
             ('scaler', StandardScaler()), ])
        else:
            pipe = Pipeline([('scaler', StandardScaler()), ])
        pipe.fit(features)

        train_features = pipe.transform(features)
        test_features = pipe.transform(features)

        score_list = []
        for i in range(0, param_vals["forest_size"]):
            (inbag_indices, outbag_indices) = sample(features, labels, bag_ids, stratified = True, sample_rate = 0.8)      

            inbag_features = features[inbag_indices]
            inbag_labels = labels[inbag_indices]
            inbag_bag_ids = bag_ids[inbag_indices]

            outbag_features = features[outbag_indices]
            outbag_labels = labels[outbag_indices]
            outbag_bag_ids = bag_ids[outbag_indices]

            model = PrototypeTreeClassifier(
                max_depth=param_vals["max_depth"], 
                min_samples_leaf=param_vals["min_samples_leaf"],
                min_samples_split=2
            )

            model.fit(inbag_features, inbag_labels, inbag_bag_ids)
            preds = model.predict(outbag_features, outbag_bag_ids)

            score = metrics.roc_auc_score(outbag_labels, preds)
            score_list.append(score)

        mean_score = sum(score_list)/len(score_list)
        key = frozenset(param_vals.items())
        param_vals_scores[key] = mean_score

    return param_vals_scores

def split_features_labels_bags(data):
    features = data[data.columns[~data.columns.isin([0, 1])]].to_numpy()
    labels = data[0].to_numpy()
    bag_ids = data[1].to_numpy()

    sort_index = np.argsort(bag_ids)
    bag_ids = bag_ids[sort_index]
    features = features[sort_index]
    
    return (features, labels, bag_ids)

def train_test_split(dataset, rep, fold, explained_variance, fit_on_full = False, custom=False):
    data = pd.read_csv(f"./datasets/{dataset}.csv", header=None)
    testbags =  pd.read_csv(f"./datasets/{dataset}.csv_rep{rep}_fold{fold}.txt", header=None)
    
    if custom:
        min_limit = testbags.min()[0]
        max_limit = testbags.max()[0]
        size = testbags.size
        size_pos = size // 2
        pos = list(range(min_limit, min_limit + size_pos))
        neg = list(range(max_limit - size_pos + 1, max_limit + 1))
        testbags = pd.DataFrame([*pos, *neg])
          
    train_data = data[~data[1].isin(testbags[0].tolist())]    
    test_data = data[data[1].isin(testbags[0].tolist())]
    
    (train_features, train_labels, train_bag_ids) = split_features_labels_bags(train_data)
    (test_features, test_labels, test_bag_ids) = split_features_labels_bags(test_data)
    
    if explained_variance < 1:
        pipe = Pipeline([('pca', PCA(n_components = explained_variance, 
                         svd_solver = "full")), 
         ('scaler', StandardScaler()), ])
    else:
        pipe = Pipeline([('scaler', StandardScaler()), ])
    
    if fit_on_full:
        pipe.fit(data[data.columns[~data.columns.isin(['0','1'])]].to_numpy())
    else:
        pipe.fit(train_features)

    train_features = pipe.transform(train_features)
    test_features = pipe.transform(test_features)
    
    return (
        train_features, 
        train_labels, 
        train_bag_ids,
        test_features, 
        test_labels,
        test_bag_ids)


class PrototypeForest:
    def __init__(self, size,
                feature_types = ["min", "mean", "max"],
                max_depth = 8, 
                min_samples_leaf = 2, 
                min_samples_split = 2, 
                prototype_count = 1,
                use_prototype_learner = True,
                early_stopping_round = 10):
        self.size = size
        self._trees = []
        self._tuning_trees = []
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.prototype_count = prototype_count
        self.use_prototype_learner = use_prototype_learner
        self.early_stopping_round = early_stopping_round
        
    def sample(self, features, labels, bag_ids):
        ids, index  = np.unique(bag_ids, return_index=True)
        group_min = np.minimum.reduceat(labels, index)
        pos_bag_size = math.ceil(np.where(group_min == 1)[0].shape[0] * 0.8)
        neg_bag_size = math.ceil(np.where(group_min == 0)[0].shape[0] * 0.8)
        bags_pos = np.random.choice(np.where(group_min == 1)[0], pos_bag_size, replace=False)
        bags_neg = np.random.choice(np.where(group_min == 0)[0], neg_bag_size, replace=False)
        df = pd.DataFrame(np.concatenate([train_bag_ids.reshape(train_bag_ids.shape[0],1),
                                          train_labels.reshape(train_labels.shape[0],1)],
                                         axis=1))
        indices_pos = df[df[0].isin(bags_pos)].index.to_numpy()
        indices_neg = df[df[0].isin(bags_neg)].index.to_numpy()
        inbag_indices = np.concatenate((indices_pos, indices_neg))
        oo_bag_mask = np.ones(labels.shape[0], dtype=bool)
        oo_bag_mask[inbag_indices] = False
        outbag_indices = np.where(oo_bag_mask == 1)
        
        return inbag_indices, outbag_indices
    
    def fit(self, features, labels, bag_ids):
        for i in range(self.size):
            if self.use_prototype_learner:
                print(f"Tree {i} will be trained")

            (inbag_indices,
             outbag_indices) = self.sample(features, labels, bag_ids)
            inbag_features = features[inbag_indices]
            inbag_labels = labels[inbag_indices]
            inbag_bag_ids = bag_ids[inbag_indices]
            tree = PrototypeTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                prototype_count = self.prototype_count,
                use_prototype_learner = self.use_prototype_learner,
                train_features = inbag_features,
                early_stopping_round = self.early_stopping_round
            )
            while True:
                try:
                    tree.fit(inbag_features, inbag_labels, inbag_bag_ids)
                except:
                    continue
                break
            self._trees.append(tree)
            
    def predict(self, features, bag_ids):
        temp = [t.predict(features, bag_ids) for t in self._trees]
        preds = np.transpose(np.array(temp))
        return mode(preds,1)[0]
    
    def predict_proba(self, features, bag_ids):
        temp = [t.predict(features, bag_ids) for t in self._trees]
        preds = np.transpose(np.array(temp))
        return np.sum(preds==1, axis=1)/self.size
    
import time

def generate_random(lower, upper):
    random_number = random.random()
    random_number = random_number + lower
    random_range = upper - lower
    random_number = random_number*random_range
    return random_number

parameters = [[0.00001, 0.05], [0.00001, 0.05],[0.00001, 0.05], [0.00001, 0.05], [0.00001, 0.05], [1],[1]]

groups = pd.read_csv("./dataset_groups.csv")

import sys
group_id = sys.argv[1]
print(f"Group id is {group_id}")

datasets = groups[groups["Group"] == int(group_id)]["dataset"].to_list()
best_params = pd.read_csv("./best_params.csv")

for dataset in datasets:
    scores = []
    info_list = []

    PCA_vals = best_params[best_params["dataset"] == dataset]["PCA"].values.tolist()
    best_depth = best_params[best_params["dataset"] == dataset]["max_depth"].values[0]
    best_size = best_params[best_params["dataset"] == dataset]["ntree"].values[0]

    if(len(PCA_vals[0]) > 1):
        PCA_vals = PCA_vals[0].split("-")
        PCA_vals = [float(x) for x in PCA_vals]
    else:
        PCA_vals = best_params[best_params["dataset"] == dataset]["PCA"].values[0]
    
    if(len(PCA_vals) > 0):
        for k in PCA_vals:
            (train_features,
                    train_labels,
                    train_bag_ids,
                    test_features,
                    test_labels,
                    test_bag_ids) = train_test_split(dataset, 5, 10, k, fit_on_full = False, custom=True)

            model = PrototypeForest(size=best_size,
                                    max_depth=best_depth,
                                    min_samples_leaf=2,
                                    min_samples_split=4,
                                    prototype_count=1,
                                    early_stopping_round= 3,
                                    use_prototype_learner = False)

            model.fit(train_features, train_labels, train_bag_ids)

            probas = model.predict_proba(test_features, test_bag_ids)

            score = metrics.roc_auc_score(test_labels, probas)
            scores.append([k, score])
    
            df = pd.DataFrame(scores, columns = ["variance","score"])

            best_row = df.iloc[df["score"].argmax()]
            best_var = best_row.get("variance")
    else:
        best_var = PCA_vals


    all_accuracy = []

    print(f"Best size is {best_size} and best depth is {best_depth} and best var is {best_var} for dataset {dataset}")
    
    for i in range(1,6):
        for j in range(1, 11):
            print(f"Rep {i}, fold {j}")
            start_time = time.time()

            (train_features,
                 train_labels,
                 train_bag_ids,
                 test_features,
                 test_labels,
                 test_bag_ids) = train_test_split(dataset, i, j, best_var, fit_on_full = False)

            model = PrototypeForest(size=best_size,
                                    max_depth=best_depth,
                                    min_samples_leaf=40,
                                    min_samples_split=80,
                                    prototype_count=1,
                                    early_stopping_round= 5,
                                    use_prototype_learner = True)

            model.fit(train_features, train_labels, train_bag_ids)

            probas = model.predict_proba(test_features, test_bag_ids)

            score = metrics.roc_auc_score(test_labels, probas)
            end_time = time.time()
            info_list_row = [dataset, i, j, best_size, best_depth, best_var, score, end_time - start_time]
            info_list.append(info_list_row)
            print(f"Score is {score}")
            all_accuracy.append(metrics.roc_auc_score(test_labels, probas))

    perf_df = pd.DataFrame(info_list, columns=["dataset", "rep", "fold", "best_size", "best_depth", "best_var",  "auc", "time"])
    perf_df.to_csv(f"./performance/{dataset}.csv")
