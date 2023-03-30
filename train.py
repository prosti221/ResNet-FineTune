import torch
import numpy as np
import torchvision.models
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, precision_score

class FineTune:
    def __init__(self, params):
        self.model = params['model']
        self.class_n = params['class_n']
        self.device = params['device']
        self.model_path = params['model_path']
        # Datasets
        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']
        # Hyperparameters
        self.epochs = params['epochs']
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.scheduler = params['scheduler'] 
        
        # Changing the last fully connected layer for class_n outputs
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.class_n)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize scheduler
        if self.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.2)
        elif self.scheduler == 'WarmRestart':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 8, T_mult = 1, eta_min = 1e-4)
        else:
            self.scheduler = None
          
    def train(self):
        best_acc = 0
        writer = SummaryWriter()
        for epoch in range(self.epochs):
            # Evaluation on eval set
            eval_metrics = self.evaluate(self.val_loader)
            # Training on train set
            self.model.train(True)
            running_loss = total = correct = 0
            step = 1
            y_true = []
            y_pred = []
            for batch_idx, data in enumerate(self.train_loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                self.optimizer.zero_grad()
                out = self.model(inputs)
                loss = self.loss_fn(out, labels)
                
                loss.backward()
                self.optimizer.step()
                total += labels.size(0)
                
                with torch.no_grad():
                    running_loss += loss.item()
                    _, predictions = torch.max(out, dim=1)
                    correct += (predictions == labels).sum().item()
                
                y_true += labels.tolist()
                y_pred += predictions.tolist()
                
                step += 1

            with torch.no_grad():
                # Add Training metrics to TensorBoard
                target_names = [f'class {i}' for i in range(self.class_n)]
                report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
                class_acc = [accuracy_score(y_true=[y_true[i] for i in range(len(y_true)) if y_true[i] == j], y_pred=[y_pred[i] for i in range(len(y_pred)) if y_true[i] == j]) for j in range(self.class_n)]
                mean_class_acc = sum(class_acc) / self.class_n
                writer.add_scalar('Training Loss', running_loss/total, epoch)
                writer.add_scalar('Training Accuracy', correct/total, epoch)
                writer.add_scalar('Training Mean Class Accuracy', mean_class_acc, epoch)
                # Add validation metrics to TensorBoard
                writer.add_scalar('Validation Loss', eval_metrics[1], epoch)
                writer.add_scalar('Validation Accuracy', eval_metrics[0], epoch)
                writer.add_scalar('Validaiton Mean Class Accuracy', eval_metrics[2], epoch)
                writer.add_scalar('Validation Mean Class Precision', eval_metrics[3], epoch)
                for i in range(len(eval_metrics[4])):
                    writer.add_scalar('Class Accuracy/' + str(i), eval_metrics[4][i], epoch)
                    writer.add_scalar('Class Precision/' + str(i), eval_metrics[5][i], epoch)
            # Save best model based on validation
            if eval_metrics[0] > best_acc:
                best_acc = eval_metrics[0]
                torch.save({
                    'epoch': epoch,
                    'step' : step,
                    'model_state_dict': self.model.state_dict(),
                    'scheduler' : self.scheduler,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, self.model_path)
                
            if self.scheduler != None:
                self.scheduler.step()
        
            print(f"\nEpoch {epoch+1}/{self.epochs}:")
            print(f"Training Loss: {running_loss/total:.4f}, Training Accuracy: {correct/total:.4f}")        
        self.model.train(False)
        writer.close()
    
    def evaluate(self, dataloader, TEST=False, PATH='./output.pt'):
        self.model.eval()
        running_loss = total = 0
        y_true = []
        y_pred = []
        if TEST:
            output = [] #(outputs, predictions, filenames) 
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                filenames = data[2]

                out = self.model(inputs)
                loss = self.loss_fn(out, labels)
                _, predictions = torch.max(out, dim=1)
                
                total += labels.size(0)
                running_loss += loss.item()
                
                y_true += labels.tolist()
                y_pred += predictions.tolist()
                
                if TEST:
                    output.append((out, predictions, filenames))
            
        target_names = [f'class {i}' for i in range(self.class_n)]
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        acc = report['accuracy']
        mean_loss = running_loss / total
        # Per class accuracy and mean class accuracy
        class_acc = [accuracy_score(y_true=[y_true[i] for i in range(len(y_true)) if y_true[i] == j], y_pred=[y_pred[i] for i in range(len(y_pred)) if y_true[i] == j]) for j in range(self.class_n)]
        mean_class_acc = np.mean(class_acc)
        # AP and mAP
        binary_class_labels = [np.array([int(y_true[i] == j) for i in range(len(y_true))]) for j in range(self.class_n)]
        binary_class_preds = [np.array([int(y_pred[i] == j) for i in range(len(y_pred))]) for j in range(self.class_n)]
        class_pr = [precision_score(binary_class_labels[i], binary_class_preds[i]) for i in range(self.class_n)]        
        mean_class_pr = np.mean(class_pr)
        
        if TEST:
            torch.save(output, PATH)
        
        print(f"\n{'Test' if TEST else 'Validation'} set metrics:")
        print("===================\n")
        print(f'Accuracy: {acc:.4f}')
        print(f'Mean Loss: {mean_loss:.4f}')
        print(f'Mean Class Accuracy: {mean_class_acc:.4f}')
        print(f'Mean Class Precision: {mean_class_pr:.4f}')
        print(f'Class Precision: {class_pr}')

        return acc, mean_loss, mean_class_acc, mean_class_pr, class_acc, class_pr
    
    
    def load_final_model(self, PATH):
        data = torch.load(PATH, map_location=torch.device(self.device))
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        
    def reproduce(self, dataloader, PATH): # PATH to old run
        outputs = torch.load(PATH,  map_location=torch.device(self.device))
        self.model.eval()
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                filenames = data[2]

                out = self.model(inputs)
                _, predictions = torch.max(out, dim=1)
                
                assert torch.equal(out, outputs[batch_idx][0]) and torch.equal(predictions, outputs[batch_idx][1]) and filenames == outputs[batch_idx][2]
        print('All the tensors are equal')
        
    def best_and_worst(self, k=10,classes=[1, 2, 3], PATH=None, plot=True): # Path is to file of test-set outputs
        assert PATH is not None
        outputs = torch.load(PATH,  map_location=torch.device(self.device))
        collapsed = [[], [], []]
        for idx, (outs, preds, filenames) in enumerate(outputs):
            if idx == 0:
                collapsed[0] = outs; collapsed[1] = preds; collapsed[2] = filenames
            else:
                collapsed[0] = torch.cat((collapsed[0], outs))
                collapsed[1] = torch.cat((collapsed[1], preds))
                collapsed[2] += filenames

        filtered = [(out, pred.item(), file) for out, pred, file in zip(*collapsed) if pred.item() in classes]
        sorted_out = sorted(filtered, key=lambda x: torch.max(x[0]).item(), reverse=True)
        grouped_by_class = {}
        for out, pred, file in sorted_out:
              grouped_by_class.setdefault(pred, []).append((torch.max(out).item(), file))
        results = {cl:{'best':vals[:k], 'worst':vals[-k:]} for cl, vals in grouped_by_class.items()}
        
        for key, bw in results.items():
            images=[]
            fig, axs = plt.subplots(2, k, figsize=(30, 6))
            fig.suptitle(f'Class {key}: Best and worst {k} images')
            for j in range(k):
                b_img = Image.open(bw['best'][j][1]).convert('RGB')
                w_img = Image.open(bw['worst'][j][1]).convert('RGB')
                b_val = bw['best'][j][0]
                w_val = bw['worst'][j][0]
                axs[0, j].imshow(b_img,interpolation='nearest', aspect='auto')
                axs[0, j].set_title(f'Value: {b_val:.2f}')
                axs[1, j].imshow(w_img,interpolation='nearest', aspect='auto')
                axs[1, j].set_title(f'Value: {w_val:.2f}')
                for ax in axs.flat:
                    ax.set(xticks=[], yticks=[])
            if plot:
                plt.show()

    def feat_map_stats(self, dataloader, image_n=200):
        non_pos_results = {}
        msd_results = {} 
        def activation(name):
            def hook(module, input, output):
                nonlocal non_pos_results, msd_results, image_n
                output.detach()
                with torch.no_grad():
                    batch_size = output.shape[0]
                    # Compute mean of non-positive values
                    avg_per_img = [(torch.sum((output[i] < 0)) / output[i].view(-1).shape[0]).item() for i in range(batch_size)]
                    avg = sum(avg_per_img) / len(avg_per_img)
                    # Compute mean of all spatial dimensions
                    msd_per_img = output.mean(dim=[2,3])
                    # Append to results
                    if module.batch_idx != 0:
                        non_pos_results[name] += (avg - non_pos_results[name]) / module.batch_idx
                        msd_results[name] = torch.cat((msd_results[name], msd_per_img), dim=0)
                    else:
                        non_pos_results[name] = avg
                        msd_results[name] = msd_per_img                
            return hook
        # List of the modules to register hooks for
        modules = [('first_conv',self.model.relu), ('layer1', self.model.layer1), ('layer2', self.model.layer2), 
                   ('layer3', self.model.layer3), ('layer4', self.model.layer4)]
        # List of hooks for the modules
        hooks = [modules[i][1].register_forward_hook(activation(modules[i][0])) for i in range(len(modules))]  
        images = 0
        self.model.eval()
        for batch_idx, data in enumerate(dataloader):
            for nam, mod in self.model.named_modules():
                mod.batch_idx = batch_idx
            inp = data[0].to(self.device)
            out = self.model(inp)
            images += inp.shape[0]
            if images >= image_n:
                break
        return non_pos_results, msd_results
    
    def plot_eigen_values(self, dataloader, dataset_name, save=False, PATH='./plots', plot=True):
        _, msd_results = self.feat_map_stats(dataloader)
        mean_per_layer = {key:value.mean(dim=0) for (key, value) in msd_results.items()}
        msd_centered = {key:value - mean_per_layer[key] for (key, value) in msd_results.items()}
        cov = {key:torch.matmul(msd_centered[key].t(), msd_centered[key]) / (value.shape[0] - 1) for (key, value) in msd_results.items()}
        eigen = {key:np.linalg.eig(value.cpu()) for (key, value) in cov.items()}
        eigen_sorted = {key:(eig_vals[np.argsort(eig_vals)[::-1]], eig_vecs[:, np.argsort(eig_vals)[::-1]]) for key, (eig_vals, eig_vecs) in eigen.items()}
        num_plots = len(eigen_sorted)
        num_rows = int(np.sqrt(num_plots))
        num_cols = int(np.ceil(num_plots / num_rows))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10*num_cols, 8*num_rows))
        axs = axs.flatten()
        for i, (key, (eig_vals, eig_vecs)) in enumerate(eigen_sorted.items()):
            axs[i].plot(eig_vals)
            axs[i].set_xlabel('Eigenvalue Index')
            axs[i].set_ylabel('Eigenvalue')
            axs[i].set_title('Eigenvalues for Key: {}'.format(key))
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
        if plot:
            plt.show()
        if save:
            fig.savefig(os.path.join(PATH, '{}_eigenvalues.png'.format(dataset_name)))
