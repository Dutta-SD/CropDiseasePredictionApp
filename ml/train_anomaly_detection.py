from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from app.anomaly import DiseaseOODModule
from app.config import ModelConfig
from app.data import ImageDataModule

ckpt_callback = ModelCheckpoint(
    dirpath="logs/PlantDiseaseOODModel",
    filename="ood_model" + "_{epoch:02d}_{VL:.2f}",
    save_top_k=1,
    mode="max",
    monitor=ModelConfig.VAL_LOSS,
)

tqdm_callback = TQDMProgressBar(refresh_rate=10)

datamodule = ImageDataModule(
    train_path=ModelConfig.TRAIN_DATA_PATH,
    val_path=ModelConfig.VAL_DATA_PATH,
    test_path=ModelConfig.TEST_DATA_PATH,
    batch_size=ModelConfig.BATCH_SIZE,
    img_size=ModelConfig.IMG_SIZE,
)

l_module = DiseaseOODModule()

seed_everything(42)

trainer = Trainer(
    max_epochs=100,
    callbacks=[ckpt_callback, tqdm_callback],
    accelerator="gpu",
    num_sanity_val_steps=2,
    default_root_dir="logs",
)


if __name__ == "__main__":
    trainer.fit(model=l_module, datamodule=datamodule)
