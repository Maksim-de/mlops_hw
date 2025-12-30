from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.abspath(__file__))
from models.joint_model import *
from data.preprocess_data import *


def train_model(
    df_path="../data/data.csv",
    little_filter_count=3,
    train_ratio=0.8,
    image_dir="../data/train_images/",
):
    # 2. Инициализируем модель
    model = MultiModalLightningModule(
        lr_image=1e-4,
        lr_text=5e-5,
        lr_joint=1e-3,
        weight_decay=1e-5,
        alpha=2.0,
        beta=100.0,
        base=0.7,
        retrieval_k_values=[1, 10, 50],
    )

    datamodule = preprocess_data_for_model(
        df_path, little_filter_count, model.tokenizator, train_ratio, image_dir
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision@50",
        mode="max",
        save_top_k=3,
        filename="best-{epoch:02d}-{val_precision@50:.4f}",
        save_last=True,
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir="./logs", name="multimodal_model", default_hp_metric=False)

    trainer = Trainer(
        max_epochs=3,
        # devices=1 if torch.cuda.is_available() else None,
        # accelerator="cpu",  # Явно указываем CPU
        accelerator='auto',
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        deterministic=False,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        # Добавляем для отладки
        num_sanity_val_steps=2,  # проверяем 2 валидационных шага перед обучением
    )

    trainer.fit(model, datamodule=datamodule)

    best_model = MultiModalLightningModule.load_from_checkpoint(checkpoint_callback.best_model_path)

    return best_model


if __name__ == '__main__':
    # Отключаем multiprocessing для диагностики
    os.environ["OMP_NUM_THREADS"] = "1"
    
    model = train_model()