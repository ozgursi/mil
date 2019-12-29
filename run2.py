import GPyOpt
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import plot_prototypes
from model import ShapeletGenerator, pairwise_dist
from mil import get_data
import matplotlib.pyplot as plt
from pandas import DataFrame
import time
from os import listdir
from os.path import isfile, join
parameters_list = []
onlyfiles = [f for f in listdir(r"./gpy") if isfile(join(r"./gpy", f))]
data_names = [i.split('.', 1)[0] for i in onlyfiles]
#data_names = ["CorelSkiing", "CorelSunset", "CorelWaterfalls"]
for data_s in data_names:
 def experiment_fn( parameters,
                    dataset=data_s,
                    n_classes=2,
                    folder=r"..\datasets",
                    n_rep=11,
                    n_fold=11,
                    n_epochs=100,
                    batch_size=1,
                    display_every=5,
                    ):  

   """
   :param dataset: Dataset name
   :param n_classes: Number of classes in the dataset
   :param folder: Folder of the dataset files
   :param n_rep: Number of reps
   :param n_fold: Number of folds
   :param n_epochs: Number of epochs
   :param batch_size: Batch size (currently only 1 is supported)
   :param n_prototypes: Number of prototypes to be learnt
   :param display_every: Display the loss and accuracy every K iterations
   :param lr_prot: Learning rate for prototypes
   :param lr_weights: Learning rate for weights
   :param reg_w: L1 or L2 regularization for weights
   :param reg_lambda_dist: Regularization coefficient for distances
   :param reg_lambda_w: Regularization coefficient for linear layer weights
   :return:
   """
   final_vals = []
   parameters = parameters[0]
   reg_lambda_dist = parameters[0]
   reg_lambda_w = parameters[1]
   reg_lambda_p = parameters[2]
   lr_prot = parameters[3]
   lr_weights = parameters[4]
   reg_w = parameters[5]
   n_prototypes = parameters[6]
   n_prototypes = n_prototypes*2
   for rep in range(n_rep-1, n_rep):
     vals = []
     for fold in range(n_fold-1, n_fold):
       accs = [] 

       use_cuda = False
       print(folder, dataset, rep, fold)
       bags_train, labels_train, bags_test, labels_test = get_data(
           folder, dataset, rep + 1, fold + 1) 

       bag_size = bags_train[0][0].shape[0]
       step_per_epoch = len(bags_train)
       lr_step = (step_per_epoch * 40)
       display = (step_per_epoch * display_every)
       max_steps = n_epochs * step_per_epoch

       model = ShapeletGenerator(n_prototypes, bag_size, n_classes)

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

       for i in range(max_steps):
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

         # N_prot x N_prot
         #M_prot_norm = torch.mm(prot_norms.transpose(0, 1), prot_norms)
         #cos_loss = torch.bmm(prototypes, prototypes.transpose(1,2)).squeeze(0)
         #cos_loss = cos_loss/M_prot_norm
         #cos_norm = cos_loss.norm(dim=0).sum() 

         # cos_loss = pd(model.prototypes, model.prototypes).sum()

         #weight_reg = model.linear_layer.weight.norm(p=1).sum()

         prototypes_pairwise = pairwise_dist(model.prototypes, model.prototypes)
         reg_prototypes = prototypes_pairwise.sum()

         weight_reg = 0
         for param in model.linear_layer.parameters():
           weight_reg += param.norm(p=reg_w).sum()

         reg_loss = reg_lambda_w*weight_reg + reg_lambda_dist*distances.sum() + reg_prototypes*reg_lambda_p
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
             print("Step : ", str(i + 1), "Loss: ",
                   total_loss.item() / display, " accuracy: ", correct / (display))
             train_loss_hist.append(total_loss.item() / display)
             train_acc_hist.append(correct / display)
             total_loss = 0
             correct = 0
             model = model.eval()
             e_loss = 0
             e_acc = 0
             y_true = []
             y_score = []
             for i in range(len(bags_test)):
               batch_inp = torch.Tensor(bags_test[i])
               batch_inp = batch_inp.view(1, batch_inp.shape[0],
                                          batch_inp.shape[1])
               targets = torch.Tensor([labels_test[i]]).type(torch.int64)
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

             print("Eval Loss: ", e_loss / len(bags_test),
                   " Eval Accuracy:", e_acc / len(bags_test), " AUC: ",
                   roc_auc_score(y_true, y_score))
             eval_loss_hist.append(e_loss / len(bags_test))
             eval_acc_hist.append(e_acc / len(bags_test))
             eval_aucs.append(roc_auc_score(y_true, y_score))
             accs.append(e_acc / len(bags_test))
             step_hist.append(i+1)
             model = model.train()
       print(str(rep), " ", str(fold), " Final Best AUC: ",
             np.max(np.array(eval_aucs)))
       end_time = time.time()
       total_time = end_time - start_time
       time_hist.append([total_time]*len(accs))
       output_data = np.column_stack([step_hist, train_loss_hist,train_acc_hist, eval_loss_hist, eval_acc_hist, eval_aucs])
       df = DataFrame(output_data, columns = ["step_hist", "train_loss_hist","train_acc_hist", "eval_loss_hist", "eval_acc_hist", "eval_aucs"])
       df['dataset'] = dataset
       df['fold'] = fold
       df['rep'] = rep
       df['time_hist'] = total_time
       export_csv = df.to_csv (r'.\export_dataframe.csv', index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path
       vals.append(np.max(np.array(eval_aucs)))
       prototypes = model.prototypes.squeeze(0).detach().numpy()
       figure_file = "shapelets_" + dataset + "_run_" + str(0) + "_" + str(
           rep) + "_" + str(fold) + ".png"
       files = "{}_{}_run_{}_{}_{}.png"
       loss_file = files.format("loss", dataset, "0", str(rep), str(fold))
       accuracy_file = files.format("acc", dataset, "0", str(rep), str(fold))

       plt.plot(train_loss_hist, label="train_loss")
       plt.plot(eval_loss_hist, label="eval_loss")
       plt.title("Loss History")
       plt.legend()
       plt.savefig(loss_file)
       plt.close()
       plt.plot(train_loss_hist, label="train_loss")
       plt.title("Only Training Loss History")
       plt.legend()
       plt.savefig("only_train_"+loss_file)
       plt.close()
       plt.plot(train_acc_hist, label="train_accuracy")
       plt.plot(eval_acc_hist, label="eval_accuracy")
       plt.title("Accuracy History")
       plt.legend()
       plt.savefig(accuracy_file)
       plt.close()
       plot_prototypes(prototypes, savefile=figure_file)

     final_vals.append(vals)
   print(np.mean(final_vals), "mean final vals")
   return np.mean(final_vals)


 #experiment_fn(n_epochs=20, display_every=0.2)

 BOUNDS = [
     {'name': 'reg_lambda_dist',
      'type': 'continuous',
      'domain': (0.0005, 0.005)},
     {'name': 'reg_lambda_w',
      'type': 'continuous',
      'domain': (0.005, 0.05)},
     {'name': 'reg_lambda_p',
      'type': 'continuous',
      'domain': (0.00005, 0.0005)},
     {'name': 'lr_prot',
      'type': 'continuous',
      'domain': (0.00001, 0.0001)},
     {'name': 'lr_weights',
      'type': 'continuous',
      'domain': (0.00001, 0.0001)},
     {'name': 'reg_w',
      'type': 'discrete',
      'domain': (1, 2)},
     {'name': 'n_prototypes',
      'type': 'discrete',
      'domain': (2, 6)}  # will be x2. ie if 2, then number of prototypes will actually be 4, if 4 then 8, etc.
 ]
 np.random.seed(777)
 optimizer = GPyOpt.methods.BayesianOptimization(
         f=experiment_fn, domain=BOUNDS,
         acquisition_type='MPI',
         acquisition_par=0.3,
         exact_eval=True,
         maximize=True
     )
 max_iter = 2
 optimizer.run_optimization(max_iter, max_time=120)
 optimizer.plot_convergence(filename="optimizer_bayesopt.png")
 print(optimizer.Y_best[-1])
 print(optimizer.x_opt)
 import winsound
 duration = 2000  # milliseconds
 freq = 1500  # Hz
 winsound.Beep(freq, duration)
 parameters = optimizer.x_opt
 parameters_output = np.transpose([parameters])
 df_parameters = DataFrame(parameters_output)
 df_parameters = df_parameters.T
 export_csv = df_parameters.to_csv (r'.\parameters_full.csv', index = None, header=False, mode='a')
 import torch
 import numpy as np
 from sklearn.metrics import roc_auc_score
 from utils import plot_prototypes
 from model import ShapeletGenerator, pairwise_dist
 from mil import get_data
 import matplotlib.pyplot as plt
 from pandas import DataFrame
 import time
 dataset=data_s
 n_classes=2
 folder=r"..\datasets"
 n_rep=5
 n_fold=10
 n_epochs=100
 batch_size=1
 display_every=5
 """
   :param dataset: Dataset name
   :param n_classes: Number of classes in the dataset
   :param folder: Folder of the dataset files
   :param n_rep: Number of reps
   :param n_fold: Number of folds
   :param n_epochs: Number of epochs
   :param batch_size: Batch size (currently only 1 is supported)
   :param n_prototypes: Number of prototypes to be learnt
   :param display_every: Display the loss and accuracy every K iterations
   :param lr_prot: Learning rate for prototypes
   :param lr_weights: Learning rate for weights
   :param reg_w: L1 or L2 regularization for weights
   :param reg_lambda_dist: Regularization coefficient for distances
   :param reg_lambda_w: Regularization coefficient for linear layer weights
   :return:
   """
 print(parameters)
 final_vals = []
 reg_lambda_dist = parameters[0]
 reg_lambda_w = parameters[1]
 reg_lambda_p = parameters[2]
 lr_prot = parameters[3]
 lr_weights = parameters[4]
 reg_w = parameters[5]
 n_prototypes = parameters[6]
 n_prototypes = n_prototypes*2
 for rep in range(n_rep):
     vals = []
     for fold in range(n_fold):
       accs = [] 

       use_cuda = False
       print(folder, dataset, rep, fold)
       bags_train, labels_train, bags_test, labels_test = get_data(
           folder, dataset, rep + 1, fold + 1)

       bag_size = bags_train[0][0].shape[0]
       step_per_epoch = len(bags_train)
       lr_step = (step_per_epoch * 40)
       display = (step_per_epoch * display_every)
       max_steps = n_epochs * step_per_epoch

       model = ShapeletGenerator(n_prototypes, bag_size, n_classes)

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

       for i in range(max_steps):
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

        # N_prot x N_prot
        #M_prot_norm = torch.mm(prot_norms.transpose(0, 1), prot_norms)
        #cos_loss = torch.bmm(prototypes, prototypes.transpose(1,2)).squeeze(0)
        #cos_loss = cos_loss/M_prot_norm
        #cos_norm = cos_loss.norm(dim=0).sum()

        # cos_loss = pd(model.prototypes, model.prototypes).sum()

        #weight_reg = model.linear_layer.weight.norm(p=1).sum()

         prototypes_pairwise = pairwise_dist(model.prototypes, model.prototypes)
         reg_prototypes = prototypes_pairwise.sum() 

         weight_reg = 0
         for param in model.linear_layer.parameters():
           weight_reg += param.norm(p=reg_w).sum()

         reg_loss = reg_lambda_w*weight_reg + reg_lambda_dist*distances.sum() + reg_prototypes*reg_lambda_p
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
             print("Step : ", str(i + 1), "Loss: ",
                   total_loss.item() / display, " accuracy: ", correct / (display))
             train_loss_hist.append(total_loss.item() / display)
             train_acc_hist.append(correct / display)
             total_loss = 0
             correct = 0
             model = model.eval()
             e_loss = 0
             e_acc = 0
             y_true = []
             y_score = []
             for i in range(len(bags_test)):
               batch_inp = torch.Tensor(bags_test[i])
               batch_inp = batch_inp.view(1, batch_inp.shape[0],
                                          batch_inp.shape[1])
               targets = torch.Tensor([labels_test[i]]).type(torch.int64)
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

             print("Eval Loss: ", e_loss / len(bags_test),
                   " Eval Accuracy:", e_acc / len(bags_test), " AUC: ",
                   roc_auc_score(y_true, y_score))
             eval_loss_hist.append(e_loss / len(bags_test))
             eval_acc_hist.append(e_acc / len(bags_test))
             eval_aucs.append(roc_auc_score(y_true, y_score))
             accs.append(e_acc / len(bags_test))
             step_hist.append(i+1)
             model = model.train()
       print(str(rep), " ", str(fold), " Final Best AUC: ",
             np.max(np.array(eval_aucs)))
       end_time = time.time()
       total_time = end_time - start_time
       time_hist.append([total_time]*len(accs))
       output_data = np.column_stack([step_hist, train_loss_hist,train_acc_hist, eval_loss_hist, eval_acc_hist, eval_aucs])
       df = DataFrame(output_data, columns = ["step_hist", "train_loss_hist","train_acc_hist", "eval_loss_hist", "eval_acc_hist", "eval_aucs"])
       df['dataset'] = dataset
       df['fold'] = fold
       df['rep'] = rep
       df['time_hist'] = total_time
       export_csv = df.to_csv (r'.\export_dataframe_runs.csv', index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path
       vals.append(np.max(np.array(eval_aucs)))
       prototypes = model.prototypes.squeeze(0).detach().numpy()
       figure_file = "shapelets_" + dataset + "_run_" + str(0) + "_" + str(
           rep) + "_" + str(fold) + ".png"
       files = "{}_{}_run_{}_{}_{}.png"
       loss_file = files.format("loss", dataset, "0", str(rep), str(fold))
       accuracy_file = files.format("acc", dataset, "0", str(rep), str(fold)) 

       plt.plot(train_loss_hist, label="train_loss")
       plt.plot(eval_loss_hist, label="eval_loss")
       plt.title("Loss History")
       plt.legend()
       plt.savefig(loss_file)
       plt.close()
       plt.plot(train_loss_hist, label="train_loss")
       plt.title("Only Training Loss History")
       plt.legend()
       plt.savefig("only_train_"+loss_file)
       plt.close()
       plt.plot(train_acc_hist, label="train_accuracy")
       plt.plot(eval_acc_hist, label="eval_accuracy")
       plt.title("Accuracy History")
       plt.legend()
       plt.savefig(accuracy_file)
       plt.close()
       plot_prototypes(prototypes, savefile=figure_file)

     final_vals.append(vals)
 print(np.mean(final_vals), "mean final vals")
 duration = 2000  # milliseconds
 freq = 1500  # Hz
 winsound.Beep(freq, duration)