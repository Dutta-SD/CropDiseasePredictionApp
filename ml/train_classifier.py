from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from app.config import ModelConfig
from app.data import ImageDataModule
from app.lm import ClassificationModule
from app.models.classification import DiseaseClassificationModel

ckpt_callback = ModelCheckpoint(
    filename="model" + "_{epoch:02d}_{VA:.2f}",
    save_top_k=1,
    mode="max",
    monitor=ModelConfig.VAL_LOSS,
)

tqdm_callback = TQDMProgressBar(refresh_rate=10)


model = DiseaseClassificationModel(ModelConfig.PRETRAINED_MODEL_NAME)

datamodule = ImageDataModule(
    train_path=ModelConfig.TRAIN_DATA_PATH,
    val_path=ModelConfig.VAL_DATA_PATH,
    test_path=ModelConfig.TEST_DATA_PATH,
    batch_size=ModelConfig.BATCH_SIZE,
    img_size=ModelConfig.IMG_SIZE,
)

l_module = ClassificationModule(
    model=model,
    num_classes=ModelConfig.NUM_OUTPUT_CLASSES,
)

seed_everything(42)
trainer = Trainer(
    max_epochs=25,
    callbacks=[ckpt_callback, tqdm_callback],
    num_sanity_val_steps=2,
)


if __name__ == "__main__":
    trainer.fit(
        model=l_module,
        datamodule=datamodule,
    )
