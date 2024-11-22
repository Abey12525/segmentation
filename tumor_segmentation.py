#! /usr/bin/env python3
import os
import gc
import h5py
import time
import torch
import random
import numpy as np
import pyopencl as cl
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler


aocx_file_path = "./conv2d.aocx"  # Path to the .aocx file
with open(aocx_file_path, 'rb') as f:
    aocx_data = f.read()

# Step 1: Set up OpenCL platform and device
platform = cl.get_platforms()[2]  # Get the first platform (can be customized)
devices = platform.get_devices(device_type=cl.device_type.ACCELERATOR)
device = devices[0]  # Select the first available FPGA accelerator device

# Step 2: Create OpenCL context and queue
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Step 5: Load the kernel and compile it
program = cl.Program(context,[device], [aocx_data]).build()

def conv2d_opencl(input_array, kernel_array, stride=1):
    # Dimensions
    input_h, input_w = input_array.shape
    kernel_h, kernel_w = kernel_array.shape
    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    # Create buffers
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
    kernel_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, output_h * output_w * np.float32().nbytes)

    # Execute kernel
    program.conv2d(queue, (output_w, output_h), None, 
                   input_buf, kernel_buf, output_buf, 
                   np.int32(input_h), np.int32(input_w),
                   np.int32(kernel_h), np.int32(kernel_w),
                   np.int32(output_h), np.int32(output_w),
                   np.int32(stride))

    # Get results
    output_array = np.empty((output_h, output_w), dtype=np.float32)
    cl.enqueue_copy(queue, output_array, output_buf)
    return output_array





ATTENTION = False
# Directory containing .h5 files
dataset_dir = "BraTS2020_training_data/content/data"

split_ratio = 0.9
split_trainset_into_chunks = 4

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['text.color']       = '#DDDDDD'



def get_filenames(print_=False):
    # Create a list of all .h5 files in the directory
    filenames = [f for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    if print_: print(f"Found {len(filenames)} .h5 files:\nExample file names:{filenames[:3]}")

    # Open the first .h5 file in the list to inspect its contents
    if filenames:
        if print_:
            file_path = os.path.join(directory, h5_files[0])
            with h5py.File(file_path, 'r') as file:
                print("Keys for each file:", list(file.keys()))
                for key, value in file.items():
                    value = value[...]
                    print('')
                    print(f"Data type of {key}: {type(value)}")
                    print(f"Shape of {key}: {value.shape}")
                    print(f"Array dtype: {value.dtype}")
                    print(f"Array max val: {np.max(value)}")
                    print(f"Array min val: {np.min(value)}")
    else:
        print("No .h5 files found in the directory.")
        return

    return filenames


def display_image_channels(image, title='Image Channels (in magma cmap)', epoch=None):
    if epoch is not None:
        title = title + ' e:%02d' % epoch
    channel_names = ['T1-weighted (T1)', 'T1-weighted post contrast (T1c)', 'T2-weighted (T2)', 'Fluid Attenuated Inversion Recovery (FLAIR)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]  # Transpose the array to display the channel
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
    if epoch:
        plt.savefig('./model/' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.' + '%016.06f'%datetime.now().timestamp() + '.png', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()

def display_mask_channels_as_rgb(mask, title='Mask Channels as RGB', epoch=None, new_name=None):
    if epoch is not None:
        title = title + ' e:%02d' % epoch
    channel_names = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, ...] * 255  # Transpose the array to display the channel
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    if epoch:
        if new_name:
            plt.savefig('./model/' + new_name + '.groundtruth' + '.png', bbox_inches='tight')
        else:
            plt.savefig('./model/' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.' + '%016.06f'%datetime.now().timestamp() + '.png', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()

def overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay', epoch=None):
    if epoch is not None:
        title = title + ' e:%02d' % epoch
    t1_image = image[0, :, :]  # Use the first channel of the image
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')
    if epoch:
        plt.savefig('./model/' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.' + '%016.06f'%datetime.now().timestamp() + '.png', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()


def read_image_and_mask(abs_path):
    with h5py.File(abs_path, 'r') as file:
        image = file['image'][...]
        mask = file['mask'][...]

    # Transpose the image and mask to have channels first
    image = image.transpose(2, 0, 1)
    mask = mask.transpose(2, 0, 1)
    return image, mask


def display_test_sample(model, test_input, test_target, device, epoch=None, display_original_channels=True):
    test_input, test_target = test_input.to(device), test_target.to(device)

    # Obtain the model's prediction
    test_pred = torch.sigmoid(model(test_input))

    # Process the image and masks for visualization
    image = test_input.detach().cpu().numpy().squeeze(0)
    mask_pred = test_pred.detach().cpu().numpy().squeeze(0)
    mask_target = test_target.detach().cpu().numpy().squeeze(0)

    # Set the plot aesthetics
    plt.rcParams['figure.facecolor'] = '#171717'
    plt.rcParams['text.color']       = '#DDDDDD'

    # Display the input image, predicted mask, and target mask
    if display_original_channels:
        display_image_channels(image)
        if epoch:
            display_mask_channels_as_rgb(mask_target, title='Ground Truth as RGB' + ' e:%02d'%epoch, epoch=epoch)
        else:
            display_mask_channels_as_rgb(mask_target, title='Ground Truth as RGB')


    if epoch:
        display_mask_channels_as_rgb(mask_pred, title='Predicted Mask Channels as RGB' + ' e:%02d'%epoch, epoch=epoch)
    else:
        display_mask_channels_as_rgb(mask_pred, title='Predicted Mask Channels as RGB')


def _display_1_image():
    _, val_dataset = get_datasets()
    test_input, test_target = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))
    image, mask = test_input, test_target

    # View images using plotting functions
    display_image_channels(image)
    display_mask_channels_as_rgb(mask)
    overlay_masks_on_image(image, mask)


class BrainScanDataset(Dataset):
    def __init__(self, filenames, transform=None):
        super().__init__()

        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load h5 file, get image and mask
        abs_path = os.path.join(dataset_dir, self.filenames[idx])
        image, mask = read_image_and_mask(abs_path)

        # Adjusting pixel values for each channel in the image so they are between 0 and 255
        for i in range(image.shape[0]):    # Iterate over channels
            min_val = np.min(image[i])     # Find the min value in the channel
            image[i] = image[i] - min_val  # Shift values to ensure min is 0
            max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
            image[i] = image[i] / max_val
        
        # Convert to float and scale the whole image
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        # if self.transform:
        #     image, mask = self.transform(image, mask)
        return image, mask



class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask

class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.vflip(image), F.vflip(mask)
        return image, mask

class RandomAffine(transforms.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        super().__init__(degrees, translate, scale, shear, resample, fillcolor)

    def __call__(self, image, mask):
        params = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
        return (F.affine(image, *params, resample=self.resample, fillcolor=self.fillcolor),
                F.affine(mask, *params, resample=self.resample, fillcolor=self.fillcolor))

class RandomPerspective(transforms.RandomPerspective):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=3):
        super().__init__(distortion_scale, p, interpolation)

    def __call__(self, image, mask):
        if random.random() < self.p:
            params = self.get_params(self.distortion_scale, image.size)
            return (F.perspective(image, *params, interpolation=self.interpolation),
                    F.perspective(mask, *params, interpolation=self.interpolation))
        return image, mask

class RandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        super().__init__(degrees, resample, expand, center)

    def __call__(self, image, mask):
        angle = self.get_params(self.degrees)
        return (F.rotate(image, angle, self.resample, self.expand, self.center),
                F.rotate(mask, angle, self.resample, self.expand, self.center))

class ToTensor(transforms.ToTensor):
    def __call__(self, image, mask):
        return F.to_tensor(image), F.to_tensor(mask)

class Normalize(transforms.Normalize):
    def __call__(self, image, mask):
        return F.normalize(image, self.mean, self.std), mask

class RandomApply(transforms.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(None, p)
        self.transforms = transforms

    def __call__(self, image, mask):
        if random.random() < self.p:
            return self.transforms(image, mask)
        return image, mask

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transform():
    transform = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomApply([
            RandomAffine(degrees=20, translate=(0.05, 0.05)),
            RandomPerspective(distortion_scale=0.05, p=1),
            RandomRotation(180),
        ], p=0.2)
    ])
    return transform


def get_datasets(split=True):
    # Split the dataset into train and validation sets (90:10)
    filenames = get_filenames()

    total_length = len(filenames)
    train_length = int(total_length * split_ratio)
    val_length = total_length - train_length


    train_transform = get_transform()
    dataset = BrainScanDataset(filenames, transform=train_transform)
    if not split:
        return dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(37))
        # 23 24 28 33 37 46 58 61 is good
            # first img in valiate set:
            # 23 volume_292_slice_64.h5
            # 24 volume_214_slice_77.h5
            # 28 volume_155_slice_73.h5
            # 33 volume_74_slice_83.h5 . in test set for trained models of previous 2 days
            # 37 volume_143_slice_83.h5 . in test set for trained models of previous 2 days
            # 46 volume_348_slice_73.h5 . in test set for trained models of previous 2 days
            # 58 volume_42_slice_99.h5
            # 61 volume_53_slice_89.h5

        # 58 28 33 61 37 is best
            # 33 volume_74_slice_83.h5 . in test set for trained models of previous 2 days
            # 37 volume_143_slice_83.h5 . in test set for trained models of previous 2 days


        return train_dataset, val_dataset


# Function to count number of parameters in a model for comparisons later
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params:,}\n')



def plot_learning_curves(train_epoch_losses, val_epoch_losses, epoch=None):
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = '#333333'

    fig, axis = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training and validation loss (NaN is used to offset epochs by 1)
    axis.plot([np.nan] + train_epoch_losses, color='#636EFA', marker='o', linestyle='-', linewidth=2, markersize=5, label='Training Loss')
    axis.plot([np.nan] + val_epoch_losses,   color='#EFA363', marker='s', linestyle='-', linewidth=2, markersize=5, label='Validation Loss')

    # Adding title, labels and formatting
    axis.set_title('Training and Validation Loss Over Epochs', fontsize=16)
    axis.set_xlabel('Epoch', fontsize=14)
    axis.set_ylabel('Loss', fontsize=14)

    axis.set_ylim(0, 0.5)
    
    axis.legend(fontsize=12)
    axis.grid(True, which='both', linestyle='--', linewidth=0.5)
    if epoch:
        plt.savefig('./model/' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.' + '%016.06f'%datetime.now().timestamp() + '.png', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.draw()

####################################################


def train_model(model, config, verbose=True):
    device = config['device']

    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    limited_train_size = config['limited_train_size']

    learning_rate = config['learning_rate']
    lr_decay_factor = config['lr_decay_factor']

    train_dataset, val_dataset = get_datasets()
    if not limited_train_size:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_epoch_losses = []
    val_epoch_losses = []

    test_input, test_target = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=False)))

    print("Training...")
    epoch_start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        batch_start_time = time.time()

        # Decay learning rate
        current_lr = learning_rate * (lr_decay_factor ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        if limited_train_size:
            # Create a new subset sampler for the current epoch
            train_subset_indices = np.random.choice(len(train_dataset), limited_train_size, replace=False)
            train_sampler = SubsetRandomSampler(train_subset_indices)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, persistent_workers=True)

        # Training step
        model.train()
        batch_train_loss = 0
        total_train_samples = 0
        for train_batch_i, (train_inputs, train_targets) in enumerate(train_dataloader, start=1):

            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)

            optimizer.zero_grad()
            train_preds = model(train_inputs)
            train_batch_loss = loss_fn(train_preds, train_targets)
            train_batch_loss.backward()
            optimizer.step()

            batch_train_loss += train_batch_loss.item() * split_trainset_into_chunks / (split_ratio / (1-split_ratio))

            current_batch_size = train_inputs.size(0)
            total_train_samples += current_batch_size

            if verbose:
                print(f"\rTraining epoch: {epoch}/{n_epochs}, batch: {train_batch_i}/{len(train_dataloader)}, batch train loss: {batch_train_loss:.6f}", end='')

            if total_train_samples % 2000 == 0:
                display_test_sample(model, test_input, test_target, device, epoch=epoch, display_original_channels=False)

                config['train_epoch_losses'] = train_epoch_losses
                config['val_epoch_losses'] = val_epoch_losses
                save_checkpoint(model, config)


        train_epoch_losses.append(batch_train_loss)


        # Validation step
        print('\nEvaluating...')
        model.eval()
        batch_val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for val_batch_i, (val_inputs, val_targets) in enumerate(val_dataloader, start=1):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_preds = model(val_inputs)
                val_batch_loss = loss_fn(val_preds, val_targets)
                batch_val_loss += val_batch_loss.item()

                batch_size_i = val_inputs.size(0)
                total_val_samples += batch_size_i

                if verbose:
                    print(f"\rValidate epoch: {epoch}/{n_epochs} batch: {val_batch_i}/{len(val_dataloader)}, batch validate loss: {batch_val_loss:.6f}", end='')

        val_epoch_losses.append(batch_val_loss)

        batch_end_time = time.time()
        if verbose:
            print(f"\nEpoch {epoch} summary: lr {current_lr:.6f}, in {(batch_end_time - batch_start_time)/60:.1f} minutes, train loss: {train_epoch_losses[-1]:.6f}, validate loss: {val_epoch_losses[-1]:.6f}\n")
            try:
                display_test_sample(model, test_input, test_target, device, epoch=epoch, display_original_channels=False)

                config['train_epoch_losses'] = train_epoch_losses
                config['val_epoch_losses'] = val_epoch_losses
                save_checkpoint(model, config)
            except NameError as e:
                print(e)

    epoch_end_time = time.time()
    print(f'Total trained {(epoch_end_time - epoch_start_time)/60:.1f} minutes.')
    print("Training complete.")
    return train_epoch_losses, val_epoch_losses


# release memory:
def clean():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation
        )
    def forward(self, x):
        return self.encoder_block(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels//2, kernel_size=(3,3), stride=1, padding=1),
            activation,
            nn.Conv2d(in_channels//2, out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation
        )
    def forward(self, x):
        return self.decoder_block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Config
        in_channels  = 4   # Input images have 4 channels
        out_channels = 3   # Mask has 3 channels
        n_filters    = 32  # Scaled down from 64 in original paper
        activation   = nn.ReLU()
        
        # Up and downsampling methods
        self.downsample  = nn.MaxPool2d((2,2), stride=2)
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Encoder
        self.enc_block_1 = UNetEncoderBlock(in_channels, 1*n_filters, activation)
        self.enc_block_2 = UNetEncoderBlock(1*n_filters, 2*n_filters, activation)
        self.enc_block_3 = UNetEncoderBlock(2*n_filters, 4*n_filters, activation)
        self.enc_block_4 = UNetEncoderBlock(4*n_filters, 8*n_filters, activation)
        
        # Bottleneck
        self.bottleneck  = nn.Sequential(
            nn.Conv2d( 8*n_filters, 16*n_filters, kernel_size=(3,3), stride=1, padding=1),
            activation,
            nn.Conv2d(16*n_filters,  8*n_filters, kernel_size=(3,3), stride=1, padding=1),
            activation
        )
        
        # Decoder
        self.dec_block_4 = UNetDecoderBlock(16*n_filters, 4*n_filters, activation)
        self.dec_block_3 = UNetDecoderBlock( 8*n_filters, 2*n_filters, activation)
        self.dec_block_2 = UNetDecoderBlock( 4*n_filters, 1*n_filters, activation)
        self.dec_block_1 = UNetDecoderBlock( 2*n_filters, 1*n_filters, activation)

        # Output projection
        self.output      = nn.Conv2d(1*n_filters,  out_channels, kernel_size=(1,1), stride=1, padding=0)

        
    def forward(self, x):
        # Encoder
        skip_1 = self.enc_block_1(x)
        x      = self.downsample(skip_1)
        skip_2 = self.enc_block_2(x)
        x      = self.downsample(skip_2)
        skip_3 = self.enc_block_3(x)
        x      = self.downsample(skip_3)
        skip_4 = self.enc_block_4(x)
        x      = self.downsample(skip_4)
        
        # Bottleneck
        x      = self.bottleneck(x)
        
        # Decoder
        x      = self.upsample(x)
        x      = torch.cat((x, skip_4), axis=1)  # Skip connection
        x      = self.dec_block_4(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_3), axis=1)  # Skip connection
        x      = self.dec_block_3(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_2), axis=1)  # Skip connection
        x      = self.dec_block_2(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_1), axis=1)  # Skip connection
        x      = self.dec_block_1(x)
        x      = self.output(x)
        return x


class AttentionEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        expansion_ratio = 4
        
        self.encoder_block = nn.Sequential(
            # 7: larger kernel sizes (ConvNeXt)
            
            # groups: depthwise; expansion_ratio + 1x1: expension. combined is called the Separable convolution.
            nn.Conv2d(in_channels,                   in_channels, kernel_size=(7,7), stride=1, padding=3, groups=in_channels),
            # batch normalization
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,  expansion_ratio*out_channels, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(expansion_ratio*out_channels, out_channels, kernel_size=(1,1), stride=1),
            #########################
            nn.Conv2d(out_channels,                 out_channels, kernel_size=(7,7), stride=1, padding=3, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, expansion_ratio*out_channels, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(expansion_ratio*out_channels, out_channels, kernel_size=(1,1), stride=1),
        )
    def forward(self, x):
        return self.encoder_block(x)

class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        expansion_ratio = 4
        
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,                    in_channels, kernel_size=(7,7), stride=1, padding=3, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,    expansion_ratio*in_channels, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(expansion_ratio*in_channels,   out_channels, kernel_size=(1,1), stride=1),
            
            nn.Conv2d(out_channels,                  out_channels, kernel_size=(7,7), stride=1, padding=3, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,  expansion_ratio*out_channels, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(expansion_ratio*out_channels,  out_channels, kernel_size=(1,1), stride=1),
        )
    def forward(self, x):
        return self.decoder_block(x)

class AttentionResBlock(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU()):
        super().__init__()
        self.query_conv     = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=1)
        self.key_conv       = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=2)
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=(1,1), stride=1)
        
        self.upsample       = nn.UpsamplingBilinear2d(scale_factor=2)
        self.activation     = activation
    
    def forward(self, query, key, value):
        query = self.query_conv(query)
        key   = self.key_conv(key)
        
        combined_attention = self.activation(query + key)
        attention_map = torch.sigmoid(self.attention_conv(combined_attention))
        upsampled_attention_map = self.upsample(attention_map)
        attention_scores = value * upsampled_attention_map
        return attention_scores


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Config
        in_channels  = 4   # Input images have 4 channels
        out_channels = 3   # Mask has 3 channels
        n_filters    = 32  # Scaled down from 64 in original paper
        activation   = nn.ReLU()
        
        # Up and downsampling methods
        self.downsample  = nn.MaxPool2d((2,2), stride=2)
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Encoder
        self.enc_block_1 = AttentionEncoderBlock(in_channels, 1*n_filters, activation)
        self.enc_block_2 = AttentionEncoderBlock(1*n_filters, 2*n_filters, activation)
        self.enc_block_3 = AttentionEncoderBlock(2*n_filters, 4*n_filters, activation)
        self.enc_block_4 = AttentionEncoderBlock(4*n_filters, 8*n_filters, activation)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            # again, separable conv.
            nn.Conv2d(     8*n_filters,   8*n_filters, kernel_size=(7,7), stride=1, padding=3, groups=8*n_filters),
            nn.BatchNorm2d(8*n_filters),
            nn.Conv2d(     8*n_filters, 4*8*n_filters, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(   4*8*n_filters,   8*n_filters, kernel_size=(1,1), stride=1),

            nn.Conv2d(     8*n_filters,   8*n_filters, kernel_size=(7,7), stride=1, padding=3, groups=8*n_filters),
            nn.BatchNorm2d(8*n_filters),
            nn.Conv2d(     8*n_filters, 4*8*n_filters, kernel_size=(1,1), stride=1),
            activation,
            nn.Conv2d(   4*8*n_filters,   8*n_filters, kernel_size=(1,1), stride=1),
        )
        
        # Decoder
        self.dec_block_4 = AttentionDecoderBlock(8*n_filters, 4*n_filters, activation)
        self.dec_block_3 = AttentionDecoderBlock(4*n_filters, 2*n_filters, activation)
        self.dec_block_2 = AttentionDecoderBlock(2*n_filters, 1*n_filters, activation)
        self.dec_block_1 = AttentionDecoderBlock(1*n_filters, 1*n_filters, activation)
        
        # Output projection
        self.output      = nn.Conv2d(1*n_filters,  out_channels, kernel_size=(1,1), stride=1, padding=0)
        
        # Attention res blocks
        self.att_res_block_1 = AttentionResBlock(1*n_filters)
        self.att_res_block_2 = AttentionResBlock(2*n_filters)
        self.att_res_block_3 = AttentionResBlock(4*n_filters)
        self.att_res_block_4 = AttentionResBlock(8*n_filters)

    def forward(self, x):
        # Encoder
        enc_1 = self.enc_block_1(x)
        x     = self.downsample(enc_1)
        enc_2 = self.enc_block_2(x)
        x     = self.downsample(enc_2)
        enc_3 = self.enc_block_3(x)
        x     = self.downsample(enc_3)
        enc_4 = self.enc_block_4(x)
        x     = self.downsample(enc_4)
        
        # Bottleneck
        dec_4 = self.bottleneck(x)
        
        # Decoder
        x     = self.upsample(dec_4)
        att_4 = self.att_res_block_4(dec_4, enc_4, enc_4)  # QKV
        x     = torch.add(x, att_4)  # Add attention masked value rather than concat
        
        dec_3 = self.dec_block_4(x)
        x     = self.upsample(dec_3)
        att_3 = self.att_res_block_3(dec_3, enc_3, enc_3)
        x     = torch.add(x, att_3)  # Add attention
        
        dec_2 = self.dec_block_3(x)
        x     = self.upsample(dec_2)
        att_2 = self.att_res_block_2(dec_2, enc_2, enc_2)
        x     = torch.add(x, att_2)  # Add attention
        
        dec_1 = self.dec_block_2(x)
        x     = self.upsample(dec_1)
        att_1 = self.att_res_block_1(dec_1, enc_1, enc_1)
        x     = torch.add(x, att_1)  # Add attention
        
        x     = self.dec_block_1(x)
        x     = self.output(x)
        return x


def save_checkpoint(model, train_config):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_config': train_config
    }
    if ATTENTION:
        torch.save(checkpoint, './model/attention_unet_checkpoint_%s.pth' % datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        torch.save(checkpoint, './model/unet_checkpoint_%s.pth' % datetime.now().strftime("%Y%m%d_%H%M%S"))





def main():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Settings for training
    _standard_epochs_to_converge = 8  #1.5
    n_epochs = _standard_epochs_to_converge * split_trainset_into_chunks
    batch_size = 5*5
    initial_learning_rate = 0.001
    final_learning_rate = 0.0001
    train_config = {
        'device': device,
        'n_epochs':          n_epochs,
        'learning_rate':     initial_learning_rate,
        'batch_size': batch_size,
        'limited_train_size': int(len(get_filenames()) * split_ratio / split_trainset_into_chunks),
        'lr_decay_factor': (final_learning_rate/initial_learning_rate) ** (1/(n_epochs-1)) if n_epochs > 1 else 1
    }

    print('lr: %f lr_decay: %f' % (train_config['learning_rate'], train_config['lr_decay_factor']))


    if ATTENTION == True:
        # Create Attention UNet model
        model = AttentionUNet()
    else:
        # Create UNet model
        model = UNet()

    count_parameters(model)


    # Train model
    train_epoch_losses, val_epoch_losses = train_model(model, train_config, verbose=True)
    plot_learning_curves(train_epoch_losses, val_epoch_losses, epoch=-1)

    # plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()


