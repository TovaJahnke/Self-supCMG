N0data = preloader("dataN0.csv")
N0bin = preloaderspecial("dataN0.csv")

import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import optim, nn, utils, Tensor
#from vime_utils import mask_generator, pretext_generator, mixer
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plt


#PRETRAINING FOR BREAST CANCER DATA

import random

piris = [[15,16],[15,17],[15,18],[16,18],[21,23],[23,24],[1,26],[19,25],[20,22],[20,25]]
binary = [0,1]
        
def Extract(lst):
    #random.seed(2)
    rnd = random.choice(binary)
    return [item[rnd] for item in lst], [item[1-rnd] for item in lst]

lead, sec = Extract(piris)

def mask_generator(p_m, x, N0=True):
    mask = np.random.binomial(1., p_m, x.shape)
    
    
    
    #UNCOMMENT THIS FOR USE OF SMG
    for row in mask:
        for i, j in zip(lead,sec):
            #row[i] = row[j]                     #METHOD 1 OF CMG
            row[i] = 1 - row[j]                  #METHOD 2 OF CMG
    
    
    if N0 == True:
        for row in mask:
            for C in row[7:12]:
                if C == 1:
                    row[7:12] = [1, 1, 1, 1, 1]
                else: 
                    row[7:12] = [0,0,0,0,0]

            for T in row[15:18]:
                if T == 1:
                    row[15:18] = [1, 1, 1]
                else: 
                    row[15:18] = [0,0,0]

    return mask

def pretext_generator(m, x):
    x = x
    no, dim = x.shape  
    # randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # corrupt samples
    x_tilde = x * (1-m) + x_bar * m  
    # define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new.float(), x_tilde.float()


class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, lr=0.0001, p_m=0.3, alpha=0.1):
        super().__init__()
        self.lr = lr
        self.p_m = p_m
        self.alpha = alpha
        self.encoder_stack = nn.Sequential(
            nn.Linear(input_size, 27),
            nn.ReLU(),
            nn.Linear(27, z) #latent vector size. 
        )
        self.mask_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z, h), # Z; LATENT REPRESENATION SIZE, H; HIDDEN SIZE
            nn.ReLU(),
            nn.Linear(h, 27),
            nn.Sigmoid()
        )
        self.feature_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z, h), # Z; LATENT REPRESENATION SIZE, H; HIDDEN SIZE
            nn.ReLU(),
            nn.Linear(h, 27),
            #nn.Sigmoid()
        )

        
    def forward(self, x):
        mask = mask_generator(self.p_m, x, N0=True)
        new_mask, corrupt_input = pretext_generator(mask, x)

        z = self.encoder_stack(corrupt_input)
        mask_pred = self.mask_stack(z)
        feature_pred = self.feature_stack(z)
        #print("Generated Mask:")
        # print(mask)
        num_ones = np.count_nonzero(mask == 1)

        total_elements = mask.size


        ratio_of_ones = num_ones / total_elements

        print("Ratio of 1s:", ratio_of_ones)       #PRINT NEW P_M
        return mask_pred, feature_pred, new_mask, z

    def get_z(self, x):
        with torch.no_grad():
            _, _,_, z = self.forward(x)
        return z
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask, z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        loss = mask_loss + feature_loss*self.alpha
        
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask, z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        val_loss = mask_loss + feature_loss*self.alpha
        val_values = {'val_loss': val_loss, 'val_mask_loss': mask_loss, 'val_feature_loss': feature_loss}
        self.log_dict(val_values, prog_bar=False, on_epoch=True)
        return val_loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask, z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        test_loss = mask_loss + feature_loss*self.alpha
        test_values = {'test_loss': test_loss, 'test_mask_loss': mask_loss, 'test_feature_loss': feature_loss}
        self.log_dict(test_values, prog_bar=False, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
    
def encodertrain(dataset, kfold=False, alpha=3.0, batch_size=15, epochs=20, lr=0.0001, folds=5, seed=10, p_m=0.3, valsize=0.3, trainsize=0.7):
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='dirpath/',  # Change this to your desired checkpoint directory
    filename='encoder-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    )
        np.random.seed(seed)
        train_index, val_index = train_test_split(list(range(len(dataset))), train_size=trainsize, test_size=valsize, random_state=seed)
        train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        self_train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        self_val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        encoder = Encoder(27, 27, lr=lr, alpha=alpha, p_m=p_m)
        en_trainer = pl.Trainer(
                limit_train_batches=batch_size,
                max_epochs=epochs,
                default_root_dir="dirpath/",            #WHERE FOR THE PRETRAINING CHECKHPOINTS TO LAND, WILL USE FOR DOWNSTREAM
                callbacks=[checkpoint_callback],  
            )


        en_trainer.fit(encoder, train_dataloaders=self_train_dataloader, val_dataloaders=self_val_dataloader)
        #latent_representations, labels = get_latent_representations(encoder, self_val_dataloader)
        val_metrics = en_trainer.test(encoder, dataloaders=self_val_dataloader)
        val_loss = val_metrics[0]['test_loss']
        print("------------------------- RESULTS -------------------------")
        print(f"val_loss: {val_loss}")
        
        return val_loss

    

for i in range(5): #TRAIN 5 TIMES FOR AVERAGE 
    print(f"Training run {i+1}") 
    encodertrain(N0data, kfold=False, alpha=3.0, batch_size=400, epochs=3000, lr=0.0001, folds=5, seed=10, p_m=0.0, valsize=0.3, trainsize=0.7)


%reload_ext tensorboard
%tensorboard --logdir= #TENSORBOARD DIR

