from models.text_models import *
from  models.image_model import *
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import madgrad

from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner



class JointModel(nn.Module):
    def __init__(self, text_model, image_model):
        super().__init__()
        self.text_model = text_model
        self.tokenizator = text_model.tokenizator
        self.image_model = image_model

    def forward(self, image_input, text_input):
        image_emb = self.image_model(image_input)
        text_emb = self.text_model(text_input)
        x = torch.cat((image_emb, text_emb), dim=1)
        x = F.normalize(x, p=2, dim=1)
        return x

# Lightning модуль
class MultiModalLightningModule(pl.LightningModule):
    def __init__(
            self,
            text_model=TinyBERTWrapper(),
            image_model= SimpleMobileNetV3(),
            lr_image=1e-4,
            lr_text=5e-5,
            lr_joint=1e-3,
            weight_decay=1e-5,
            alpha=2.0,
            beta=100.0,
            base=0.7,
            retrieval_k_values=[1, 10, 50]  # Добавил K значения для метрик
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['text_model', 'image_model'])
        self.validation_step_outputs = []

        # Инициализация моделей
        self.model = JointModel(text_model, image_model)
        self.tokenizator = text_model.tokenizator
        self.criterion = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

    def forward(self, image_input, text_input):
        return self.model(image_input, text_input)

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        embeddings = self(images, texts)
        loss = self.criterion(embeddings, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels = batch
        embeddings = self(images, texts)
        loss = self.criterion(embeddings, labels)

        self.validation_step_outputs.append({
            'embeddings': embeddings.detach().cpu(),
            'labels': labels.detach().cpu(),
            'val_loss': loss.detach()
        })

        # Логируем loss
        self.log('val_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_end(self):
        # Проверяем, что есть данные
        if not self.validation_step_outputs:
            # Логируем нулевые значения
            self.log('val_loss', torch.tensor(0.0), prog_bar=True)
            for k in self.hparams.retrieval_k_values:
                self.log(f'val_recall@{k}', torch.tensor(0.0), prog_bar=(k == 50))
                self.log(f'val_precision@{k}', torch.tensor(0.0), prog_bar=(k == 50))
                self.log(f'val_f1@{k}', torch.tensor(0.0), prog_bar=(k == 50))
            self.validation_step_outputs.clear()
            return

        try:
            # Собираем данные
            all_embeddings = torch.cat([x['embeddings'] for x in self.validation_step_outputs], dim=0)
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs], dim=0)

            # Средний loss
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_loss', avg_loss, prog_bar=True)

            # Вычисляем метрики
            if len(all_embeddings) > 0 and len(all_labels) > 0:
                embeddings_np = all_embeddings.numpy()
                labels_np = all_labels.numpy().astype(int)

                for k in self.hparams.retrieval_k_values:
                    try:
                        metrics = compute_retrieval_metrics(embeddings_np, labels_np, k)

                        # Логируем все метрики
                        for metric_name, metric_value in metrics.items():
                            log_name = f'val_{metric_name}'
                            self.log(log_name, metric_value,
                                     prog_bar=(metric_name == 'recall@50'))

                    except Exception as e:
                        print(f"Ошибка при вычислении метрик для k={k}: {e}")
                        # Логируем нулевые значения
                        self.log(f'val_recall@{k}', torch.tensor(0.0), prog_bar=(k == 50))
                        self.log(f'val_precision@{k}', torch.tensor(0.0), prog_bar=False)
                        self.log(f'val_f1@{k}', torch.tensor(0.0), prog_bar=False)
            else:
                for k in self.hparams.retrieval_k_values:
                    self.log(f'val_recall@{k}', torch.tensor(0.0), prog_bar=(k == 50))
                    self.log(f'val_precision@{k}', torch.tensor(0.0), prog_bar=False)
                    self.log(f'val_f1@{k}', torch.tensor(0.0), prog_bar=False)

        except Exception as e:
            print(f"Ошибка в on_validation_epoch_end: {e}")
            # Логируем нулевые значения
            self.log('val_loss', torch.tensor(0.0), prog_bar=True)
            for k in self.hparams.retrieval_k_values:
                self.log(f'val_recall@{k}', torch.tensor(0.0), prog_bar=(k == 50))

        # Очищаем outputs
        self.validation_step_outputs.clear()

    def on_validation_epoch_start(self):
        # Очищаем outputs в начале каждой эпохи
        self.validation_step_outputs.clear()

    def validation_step_end(self, step_output):
        # Сохраняем output для дальнейшей обработки
        self.validation_outputs.append(step_output)
        return step_output

    def configure_optimizers(self):
        # Создаем группы параметров с разными learning rates
        param_groups = []

        # Параметры для image_model
        image_params = []
        for name, param in self.model.named_parameters():
            if 'image_model' in name and param.requires_grad:
                image_params.append(param)

        if image_params:
            param_groups.append({
                'params': image_params,
                'lr': self.hparams.lr_image,
                'name': 'image_params'
            })

        # Параметры для text_model
        text_params = []
        for name, param in self.model.named_parameters():
            if 'text_model' in name and param.requires_grad:
                text_params.append(param)

        if text_params:
            param_groups.append({
                'params': text_params,
                'lr': self.hparams.lr_text,
                'name': 'text_params'
            })

        # Остальные параметры (joint слои)
        other_params = []
        for name, param in self.model.named_parameters():
            if ('image_model' not in name) and ('text_model' not in name) and param.requires_grad:
                other_params.append(param)

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.hparams.lr_joint,
                'name': 'joint_params'
            })

        # Создаем один оптимизатор с разными LR для разных групп
        optimizer = madgrad.MADGRAD(
            param_groups,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer