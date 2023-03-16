# CorruptedDataLoader
Chen Liu

## Contributions
We provide a simple wrapper around PyTorch DataLoader to **intentionally mess up the input/label correspondence**.

## Story behind the scene
In the majority of times, when we train a machine learning model, we pay extra attention to make sure the inputs and labels are correctly matched. In occasional situations, however, we may want the opposite to happen. One such possibility is, as outlined in the paper ["Understanding deep learning requires rethinking generalization"](https://arxiv.org/abs/1611.03530), we may want to **corrupt the training set and overfit a model on random labels**.

Despite careful search on the internet, we were unable to find existing open-source implementations to achieve this purpose. Therefore we designed our own method to achieve this purpose and provided it to those who may have a similar need.

## Details
This repository currently only contains a single file, which itself contains a single class called `CorruptedDataLoader`. `CorruptedDataLoader` is a wrapper around a Pytorch `DataLoader`. The `Dataloader` may hold arbitrary `dataset`s, while in the current implementation, we only support the following `dataset`s:

1. `torchvision.datasets.MNIST`
2. `torchvision.datasets.CIFAR10`
3. `torchvision.datasets.CIFAR100`
4. `torchvision.datasets.STL10`

Meanwhile, it can be easily adapted to any custom `dataset`, as long as you know under what key the `labels` are stored.

## Usage
To use, simply copy `CorruptedDataLoader` to an appropriate location in your codebase and modify as you need. Don't forget to give us a `star` if you use it and find it helpful.

## Citation
To be added
