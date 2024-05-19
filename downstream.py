## downstream task auc
from pytorch_lightning.loggers import TensorBoardLogger


import matplotlib.pyplot as plt


#SAME ENCODER AND VALUES OF HYPERPARAMETERS Z AND H AS IN THE PRETRAINING OF THE CHECKPOINT
class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, lr=0.0001, p_m=0.03, alpha=3.0):
        super().__init__()
        self.lr = lr
        self.p_m = p_m
        self.alpha = alpha
        self.encoder_stack = nn.Sequential(
            nn.Linear(27, 27), 
            nn.ReLU(),
            nn.Linear(27, Z)
        )
        self.mask_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Z, H), 
            nn.ReLU(),
            nn.Linear(H, 27),
            nn.Sigmoid()
        )
        self.feature_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Z, H),
            nn.ReLU(),
            nn.Linear(H, 27),
            #nn.Sigmoid()
        )

    def forward(self, x):
        mask = mask_generator(self.p_m, x, N0=True)
        new_mask, corrupt_input = pretext_generator(mask, x)

        z = self.encoder_stack(corrupt_input)
        mask_pred = self.mask_stack(z)
        feature_pred = self.feature_stack(z)
        return mask_pred, feature_pred, new_mask,z
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask,z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        loss = mask_loss + feature_loss*self.alpha
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask,z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        val_loss = mask_loss + feature_loss*self.alpha
        val_values = {'val loss': val_loss, 'val mask loss': mask_loss, 'val feature loss': feature_loss}
        self.log_dict(val_values, prog_bar=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask_pred, feature_pred, new_mask,z = self.forward(x)
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, new_mask)
        feature_loss = nn.functional.mse_loss(feature_pred, x)
        test_loss = mask_loss + feature_loss*self.alpha
        test_values = {'test_loss': test_loss, 'test_mask_loss': mask_loss, 'test_feature_loss': feature_loss}
        self.log_dict(test_values, prog_bar=True, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


class EncoderMLP(pl.LightningModule):
    def __init__(self, trained_encoder, input_size, hidden_size, num_classes, lr=0.0001, weight_decay=0.02, en_weight_decay=0.01):
        super().__init__()
       # self.val_losses = [] #try
        self.metric = torchmetrics.functional.auroc
        self.lr = lr
        self.weight_decay = weight_decay
        self.en_weight_decay = en_weight_decay
        self.trained_encoder = trained_encoder
        self.automatic_optimization = False
        self.MLPstack = nn.Sequential(
            nn.Linear(Z, 1), #input needs to be same as latent vector
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.trained_encoder(x)
        return self.MLPstack(z)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        mlp_opt, e_opt = self.optimizers()
        mlp_opt.zero_grad()
        e_opt.zero_grad()
        self.manual_backward(loss)
        mlp_opt.step()
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_opt.step()
        values = {'training_loss': loss}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        val_auc = self.metric(pred.view(pred.shape[0]), y, task='binary')
        values = {'val_loss': loss, 'val_auc': val_auc}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        
        return val_auc

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.binary_cross_entropy(pred.view(pred.shape[0]), y.float())
        test_auc = self.metric(pred.view(pred.shape[0]), y, task='binary')
        values = {'test_loss': loss, 'test_auc': test_auc}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        return test_auc
    
    def on_train_epoch_end(self):
        mlp_sch, e_sch = self.lr_schedulers() 
        mlp_sch.step(self.trainer.callback_metrics["val_loss"])
        if self.current_epoch >= 300 and self.current_epoch <= 1000:
            e_sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        mlp_opt = torch.optim.Adam(self.MLPstack.parameters(), self.lr, weight_decay=self.weight_decay)
        e_opt = torch.optim.Adam(self.trained_encoder.parameters(), self.lr, weight_decay=self.en_weight_decay)
        mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt, factor=0.3, patience=70)
        e_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(e_opt, factor=0.3, patience=70)
        return [mlp_opt, e_opt], [mlp_scheduler, e_scheduler] 





   


def encoderMLPtrain(dataset, checkpoint, kfold=False, batch_size=15, epochs=20, lr=0.001, folds=5, seed=10, valsize=0.25, trainsize=0.75, weight_decay=0.02, en_weight_decay=0.01, encoder_type="VIME"):
    train_index, val_index = ksplit(dataset, trainsize, valsize, seed)
    train_subsampler, val_subsampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        trained_model = Encoder.load_from_checkpoint(checkpoint, input_size=27, hidden_size=27) #change with cheockpoint (sizes)
        #trained_model = trained_model.encoder_stack
        trained_encoder = trained_model.encoder_stack
        
        
        for param in trained_encoder.parameters():  #FREEZE PARAMETERS FROM ENCODER
            param.requires_grad = False
        
        encoder_mlp = EncoderMLP(trained_encoder, 27, 27, 1, lr=lr, weight_decay=weight_decay, en_weight_decay=en_weight_decay)
        trainer = pl.Trainer(
    limit_train_batches=batch_size,
    max_epochs=epochs,
    default_root_dir="MixEncoder_logs/",
    logger=TensorBoardLogger("DIR/")
)

    encoder_mlp.MLPstack.apply(weights_init)
    trainer.fit(encoder_mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    val_metrics = trainer.test(encoder_mlp, dataloaders=val_dataloader)
    val_auc = val_metrics[0]['test_auc']
    val_loss = val_metrics[0]['test_loss']
    print("------------------------- RESULTS -------------------------")
    print(f"val_loss: {val_loss}, val_auc: {val_auc}")
    return val_auc, val_loss

checkpoint = "PUT CHECKPONT FROM PRETRAINING HERE"


encoderMLPtrain(N0data, checkpoint, kfold=False, batch_size=200, epochs=5500, lr=0.0001, folds=5, seed=10, valsize=0.30, trainsize=0.70, weight_decay=0.0, en_weight_decay=0.0, encoder_type="VIME")


"""%reload_ext tensorboard
%tensorboard --logdir=DIR/ --port 8841
"""
