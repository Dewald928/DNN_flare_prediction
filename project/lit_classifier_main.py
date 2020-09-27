from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from project.datamodule import MNISTDataModule
from project.models import LitClassifier


def cli_main():
    # from project.datasets.mnist import mnist
    from torchvision.datasets import MNIST
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int)

    # optional... automatically add all the params
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # data
    dm = MNISTDataModule()

    # model
    model = LitClassifier(**vars(args))

    # training
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=2,
                         limit_train_batches=200)
    trainer.fit(model, dm)

    trainer.test(datamodule=dm)


if __name__ == '__main__':  # pragma: no cover
    cli_main()
