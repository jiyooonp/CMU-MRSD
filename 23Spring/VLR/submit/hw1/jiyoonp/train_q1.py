import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of 224x224 for the rest of the questions.

    # TODO experiment a little and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    args = ARGS(
        epochs=5,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=1e-4,
        batch_size=6,
        step_size=3,
        gamma= 0.5

    )

    print(args)

    # initializes the model
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3)
    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map

    # for lr in [0.0001, 0.00001, 0.000001, 0.00005]:
    #     for batch in [4, 8, 16]:
    #         for gamma in [0.5, 0.7]:


    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

'''
        epochs=5,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=2e-4,
        batch_size=8,
        step_size=1,
        gamma= 0.7
'''
'''
        epochs=5,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=2e-4,
        batch_size=4,
        step_size=1,
        gamma= 0.6
        map:  0.21373215736861245
test map: 0.2115234942262623
'''

'''
        epochs=5,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=1e-4,
        batch_size=6,
        step_size=2,
        gamma= 0.5
0.22791830550318665
'''

'''
        epochs=5,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=1e-4,
        batch_size=6,
        step_size=3,
        gamma= 0.5
    2105
test map: 0.23243651980444494
'''