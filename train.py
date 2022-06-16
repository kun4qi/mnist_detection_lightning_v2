import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.optim import lr_scheduler
import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import callbacks

from models import Generator
from models import Discriminator
from dataio import MNISTDataModule
from utils import load_json
from utils import anomaly_score

class ANOGAN(LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.anomaly_score = anomaly_score
        # networks
        self.generator = Generator(z_dim=config.model.z_dim, gen_filters=config.model.gen_filters)
        self.discriminator = Discriminator(input_dim=config.model.input_dim, dis_filters=config.model.dis_filters)
        self.anomalities = []

    def criterion(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        return loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train generator
        if optimizer_idx == 0:
          g_loss = self.__share_step(batch, 'train', optimizer_idx)
          tqdm_dict = {'g_loss': g_loss}
          output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'g_loss': g_loss
            })
          return output

        # train discriminator
        if optimizer_idx == 1:
          d_loss = self.__share_step(batch, 'train', optimizer_idx)
          tqdm_dict = {'d_loss': d_loss}
          output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'd_loss': d_loss
            })
          return output
          
    def validation_step(self, batch, batch_idx):
        # valid generator
        g_loss, d_loss = self.__share_step(batch, 'valid')
        return {'val_loss_g': g_loss, 'val_loss_d': d_loss}

    def __share_step(self, batch, mode, optimizer_idx=None):
        images = batch

        self.batch_size = images.size(0)

        # 真偽のラベルを定義
        label_real = torch.full((images.size(0),), 1.0).type_as(images)
        label_fake = torch.full((images.size(0),), 0.0).type_as(images)
         
        # 潜在変数から偽の画像を生成
        z = torch.randn(images.size(0), self.config.model.z_dim).view(images.size(0), self.config.model.z_dim, 1, 1).type_as(images)
        fake_images = self.generator(z)
        
        if mode == 'train' and optimizer_idx == 0:
          # Discriminator によって真偽判定
          d_out_fake, _ = self.discriminator(fake_images)

          # 損失の計算
          g_loss = self.criterion(d_out_fake.view(-1).type_as(images), label_real)

          dataset_size = 0
          running_loss = 0.0
          running_loss += (g_loss * self.batch_size)
          dataset_size += self.batch_size
          epoch_loss = running_loss / dataset_size
          self.log('train_g_loss', epoch_loss)

          return epoch_loss

        elif mode == 'train' and optimizer_idx == 1:
          # Discriminator で偽の画像と本物の画像を判定
          d_out_real, _ = self.discriminator(images)
          d_out_fake, _ = self.discriminator(fake_images)

          # 損失の計算
          d_loss_real = self.criterion(d_out_real.view(-1).type_as(images), label_real)
          d_loss_fake = self.criterion(d_out_fake.view(-1).type_as(images), label_fake)
          d_loss = d_loss_real + d_loss_fake

          dataset_size = 0
          running_loss = 0.0
          running_loss += (d_loss * self.batch_size)
          dataset_size += self.batch_size
          epoch_loss = running_loss / dataset_size
          self.log('train_d_loss', epoch_loss)

          return epoch_loss

        else: #valid
          # Discriminator によって真偽判定
          d_out_fake, _ = self.discriminator(fake_images)
          d_out_fake = d_out_fake.type_as(images)
          # 損失の計算
          g_loss = self.criterion(d_out_fake.view(-1).type_as(images), label_real)

          # valid discriminator
          # Discriminator で偽の画像と本物の画像を判定
          d_out_real, _ = self.discriminator(images)
          d_out_fake, _ = self.discriminator(fake_images)

          # 損失の計算
          d_loss_real = self.criterion(d_out_real.view(-1).type_as(images), label_real)
          d_loss_fake = self.criterion(d_out_fake.view(-1).type_as(images), label_fake)
          d_loss = d_loss_real + d_loss_fake

          return  g_loss, d_loss
    
    def validation_epoch_end(self, outputs):
        g_losses = []
        d_losses = []
        for out in outputs:
            g_loss, d_loss = out['val_loss_g'].item(), out['val_loss_d'].item()
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        g_losses = torch.tensor(g_losses, dtype=torch.float32).mean()
        d_losses = torch.tensor(d_losses, dtype=torch.float32).mean()
        
        dataset_size_g = 0
        running_loss_g = 0.0
        dataset_size_d = 0
        running_loss_d = 0.0
        running_loss_g += (g_losses * self.batch_size)
        dataset_size_g += self.batch_size
        epoch_loss_g = running_loss_g / dataset_size_g
        running_loss_d += (d_losses * self.batch_size)
        dataset_size_d += self.batch_size
        epoch_loss_d = running_loss_d / dataset_size_d
        self.log('valid_loss_g', epoch_loss_g, prog_bar=True)
        self.log('valid_loss_d', epoch_loss_d, prog_bar=True)

    def test_step(self, batch, batch_idx):
        print(f'now step {batch_idx}')
        input_images = batch
        
        self.generator.eval()
        self.discriminator.eval()

        z = torch.randn(input_images.size(0), 20).view(input_images.size(0), 20, 1, 1).type_as(input_images)
        print('search z')
        z.requires_grad = True
        z_optimizer = torch.optim.Adam([z], lr=1e-3)

        with torch.enable_grad():
          for epoch in range(5000):
            #z探し
            fake_images = self.generator(z)
            loss, _, _ = self.anomaly_score(input_images, fake_images, self.discriminator)
            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()
        
        #異常度の計算
        fake_images = self.generator(z)
        _, anomality, _ = self.anomaly_score(input_images, fake_images, self.discriminator)
        anomality = anomality.cpu().detach().numpy()
        anomality = anomality.tolist()
        self.anomalities.extend(anomality)
        return self.anomalities

    def test_epoch_end(self, anomalities):
        anomality = anomalities[0]
        df_test = pd.read_csv(self.config.dataset.root_dir_path+"mnist_test.csv", dtype = np.float32)
        df_test.rename(columns={'7': 'label'}, inplace=True)
        df_test = df_test.query("label in [1.0, 0.0]").head(500)
        df_anomality=pd.DataFrame(anomality,columns = ['anomality'])
        df_test = df_test.reset_index(drop=True)
        df=pd.concat([df_test, df_anomality],axis=1)
        
        s=df['anomality'].min()
        e=df['anomality'].max()
        #精度が上がるしきい値を探索
        list_auc = []
        for th in np.arange(s, e, 100.0):
          df=pd.concat([df_test, df_anomality],axis=1)
          df['judge'] = [1 if s > th else 0 for s in df['anomality']]
          df['label'] = [1 if s ==0 else 0 for s in df['label']] 
          aucroc = roc_auc_score(df['label'].values, df['judge'].values)
          list_auc.append((th, aucroc))
        ths,auc = sorted(list_auc, key=lambda x:x[1], reverse=True)[0]
        print(ths,auc)
        
    def configure_optimizers(self):
      optimizerG = torch.optim.AdamW(self.generator.parameters(), lr=self.config.optimizer.learning_rate, weight_decay=self.config.optimizer.weight_decay)
      optimizerD = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config.optimizer.learning_rate, weight_decay=self.config.optimizer.weight_decay)
      schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG,T_max=self.config.optimizer.T_max, eta_min=self.config.optimizer.min_lr)
      schedulerD = lr_scheduler.CosineAnnealingLR(optimizerD,T_max=self.config.optimizer.T_max, eta_min=self.config.optimizer.min_lr)
      return [optimizerG, optimizerD], [schedulerG, schedulerD]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist detection with pytorch lightning')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-t', '--test', help='run as test mode', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)
    dm = MNISTDataModule(config)
    trainer = Trainer(
      default_root_dir=config.save.output_root_dir,
      gpus=1,
      max_epochs=config.training.epochs,
      deterministic=True,
      )

    if not args.test:
      model = ANOGAN(config, *dm.size())
      trainer.fit(model, dm)

    else:
      model = ANOGAN.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config,)
      trainer.test(model, dm)