# TSM Experiments

We conducted 4 experiments on our TSM models trained from our custom code in comparison with the TSM model trained from original paper.

## Experiment 1

We removed `Stack()` transformation from both `train_loader` and `val_loader`. The shape of the dataloader is changed from `[B, N*C, H, W]` to `[B, N, C, H, W]`, and we transforms the tensor back to `[B, N*C, H, W]` in training.

```python
train_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix,
               transform=torchvision.transforms.Compose([                 
                   train_augmentation,
                   ToTorchFormatTensor(),
                   normalize,
               ]), dense_sample=args.dense_sample, split_name='train'),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True,
    drop_last=True)  # prevent something not % n_GPU
```

## Experiment 2

We added another resize and padding to the data preprocessing in `train_loader`. Originally we planned to test the effect of our method of data preprocessing, but we missed the negative impact of applying both padding and cropping to the input data.
Unlike other experiment in which we trained the model from ImageNet pretrained model for 50 epochs, we train the model in experiment 2 from the model checkpoint trained in Experiment 1.

```python
train_loader = torch.utils.data.DataLoader(
  TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
             new_length=data_length,
             modality=args.modality,
             image_tmpl=prefix,
             transform=torchvision.transforms.Compose([
                 image_resizer_and_padder,                 
                 train_augmentation,
                 ToTorchFormatTensor(),
                 normalize,
             ]), dense_sample=args.dense_sample, split_name='train'),
  batch_size=args.batch_size, shuffle=True,
  num_workers=args.workers, pin_memory=True,
  drop_last=True)  # prevent something not % n_GPU
```


## Experiment 3

We removed `GroupMultiScaleCrop()` from the data augmentation of the input image, and we want to test the impact of this augmentation on the drop of accuracy of testing.

```python
train_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix,
               transform=torchvision.transforms.Compose([
                   GroupScale(int(scale_size)),
                   GroupCenterCrop(crop_size),
                   GroupRandomHorizontalFlip(is_flow=False),
                   ToTorchFormatTensor(),
                   normalize,
               ]), dense_sample=args.dense_sample, split_name='train'),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True,
    drop_last=True)  # prevent something not % n_GPU
```


## Experiment 4

We based on the configuration of Experiment 1 and added `GroupRandomRotation()` to data augmentation, and we want to test the impact of this augmentation on the accuracy of testing.

```python
train_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
            new_length=data_length,
            modality=args.modality,
            image_tmpl=prefix,
            transform=torchvision.transforms.Compose([                     
                train_augmentation,
                GroupRandomRotation(),
                ToTorchFormatTensor(),  # from [B,H,W,C] to [B,C,H,W]
                normalize,
            ]), dense_sample=args.dense_sample, split_name='train'),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True,
    drop_last=True)  # prevent something not % n_GPU
```

## Results

To be filled in (figures + evaluation metrics for all experiments)


