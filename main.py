from dataloader import *
from train import *
from torchvision import transforms
import torch.nn as nn
import torchvision.models 

if __name__ == '__main__':
    # The directories are set to be inside the data/ folder
    root_dir = './data/mandatory1_data'
    imagenet_dir = './data/ILSVRC2012_img_val'
    cifar_dir = './data/'
    # Parameters
    batch_size = 16
    seed = 65
    num_workers = 8 
    plot = False # Set to true to plot all of the statistics
    # Reproducability stuff
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    transform = transforms.Compose(
                [transforms.Resize((224,224)), 
                 transforms.ToTensor(), 
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    # Load datasets and check class distribution across all sets
    dataset = MandatoryDataset(root_dir, transform=transform)

    train_loader, val_loader, test_loader = load_splits(dataset, batch_size, seed=seed, num_workers=num_workers)
    # Check that the datasets are disjoint
    check_disjoint(train_loader, val_loader, test_loader)
    # Fine tune parameters
    params = {'model'        : torchvision.models.resnet18(pretrained=True),
              'model_path'   : './new_model.pt',
              'class_n'      : 6,
              'device'       : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              'train_loader' : train_loader,
              'val_loader'   : val_loader,
              'epochs'       : 10,
              'lr'           : 0.00002,
              'weight_decay' : 0.001,
              'scheduler'    : 'WarmRestart'
             }

    # Fine tune, validate and reproduce
    fine_tuned_model = FineTune(params)
    #fine_tuned_model.train()

    fine_tuned_model.load_final_model('./old_model.pt')

    #testset_metrics = fine_tuned_model.evaluate(test_loader, TEST=True,  PATH='new_output')
    fine_tuned_model.best_and_worst(k=10, PATH='./old_output', plot=plot)

    # This reproduce seems to only work when training the model again on the same parameters and comparing with old saved tensor,
    # loading a model from file breaks something I haven't been able to find.
    #fine_tuned_model.reproduce(test_loader, PATH='old_output')

    # Task 2 and 3 statistics
    initialized_model = FineTune(params)


    # Statistic images for mandatory dataset
    _, _, mand_images = load_splits(dataset, batch_size=1, seed=seed, num_workers=num_workers)
    # Statistic images for ImageNet
    imagenet_images = load_ImageNet(imagenet_dir, batch_size, transform, num_workers=0)
    # Statistic images for CIFAR-100
    cifar_images = load_CIFAR(cifar_dir, batch_size, transform, download=True, num_workers=num_workers)

    # Compute and plot statistics on mandatory dataset (Set output to True to show plots)
    fine_tuned_model.plot_eigen_values(mand_images, dataset_name='mandatory_dataset', save=False, plot=plot)
    # Compute and plot statistics on ImageNet dataset
    initialized_model.plot_eigen_values(imagenet_images, dataset_name='ImageNet', save=False, plot=plot)
    # Compute and plot statistics on CIFAR-100 dataset
    initialized_model.plot_eigen_values(cifar_images, dataset_name='CIFAR', save=False, plot=plot)
