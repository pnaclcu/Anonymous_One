import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from lion_pytorch import Lion
from med_seg_diff_pytorch.med_seg_diff_pytorch import Unet, MedSegDiff
#from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
#from accelerate import Accelerator
import wandb
from utils.dataset import BasicDataset
import random
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import skimage.io as io
torch.backends.cudnn.enabled = False
## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-slr', '--scale_lr', action='store_true', help="Whether to scale lr.")
    parser.add_argument('-rt', '--report_to', type=str, default="wandb", choices=["wandb"],
                        help="Where to log to. Currently only supports wandb")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-ga', '--gradient_accumulation_steps', type=int, default=4,
                        help="The number of gradient accumulation steps.")
    parser.add_argument('-img', '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='learning rate') #default=5e-4
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.95,
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999,
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('-aw', '--adam_weight_decay', type=float, default=1e-6,
                        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument('-ae', '--adam_epsilon', type=float, default=1e-08,
                        help='Epsilon value for the Adam optimizer.')
    parser.add_argument('-ul', '--use_lion', type=bool, default=False, help='use Lion optimizer')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=3,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='./dataset', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='CAMUS', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=100, help='save_every n epochs (default: 100)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    parser.add_argument('--device', default='cuda', help='path to pt file to load from')
    parser.add_argument('--multi_device', default='0,1,2,3', help='path to pt file to load from')


    return parser.parse_args()

def acquire_all_patient(dir):
    all_patients = []
    all_group = os.listdir(dir)
    for group in all_group:
        group_path = os.path.join(dir, group)
        group_patients = os.listdir(group_path)
        for patient in group_patients:
            patient_path = os.path.join(group_path, patient)
            all_patients.append(patient_path)
    return all_patients
def acquire_all_img(temp_list):
    random.seed(42)
    full_img_path=[]
    for patient in temp_list:

        img_path=os.path.join(patient,'img')
        all_img=os.listdir(img_path)
        all_img=[os.path.join(img_path,i) for i in all_img]
        #all_mask=[i.replace('img','mask') for i in all_img]

        full_img_path.extend(all_img)
        #full_mask_path.extend(all_mask)
    full_img_path=sorted(full_img_path)

    random.shuffle(full_img_path)
    #print(full_img_path)
    full_mask_path = [i.replace('img', 'mask') for i in full_img_path]


    return full_img_path,full_mask_path


def load_data(args):
    # Load dataset
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=True,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=False)
    elif args.dataset =='RV':

        args.data_path='../RV_data_parts'
        transform_list=[transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        val_percent=0.2
        all_patient = acquire_all_patient(dir=args.data_path)
        train_patient, test_patient, _, _ = train_test_split(all_patient, all_patient, test_size=val_percent,random_state=42)
        train_img, train_mask = acquire_all_img(train_patient)
        dataset = BasicDataset(train_img, train_mask, transform=transform_train,scale=1 if args.image_size==256 else 0.5,training=True)
    elif  args.dataset =='CAMUS':
        #
        args.data_path='../CAMUS'
        transform_list=[transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        train_path=os.path.join(args.data_path,'training')
        test_path=os.path.join(args.data_path,'testing')
        train_patient=os.listdir(train_path)
        test_patient=os.listdir(test_path)
        train_patient=[os.path.join(train_path,i) for i in train_patient]
        test_patient = [os.path.join(test_path, i) for i in test_patient]
        train_img, train_mask = acquire_all_img(train_patient)
        dataset = BasicDataset(train_img, train_mask, transform=transform_train,scale=1 if args.image_size==256 else 0.5,training=True)



    #######################################################################################################
    #Note : There is no need to evaluate the performance on VALIDATION DATASET due to the traininig is slow.
    #######################################################################################################
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    return dataset


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_device

    device_num = 1 if args.multi_device == False else len(args.multi_device)
    if args.dataset=='RV':
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints_RV')
    else:
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints_camus')
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset = load_data(args)
    if device_num==1:
        device = args.device
        model = Unet(
            dim=args.dim,
            image_size=args.image_size,
            dim_mults=(1, 2, 4, 8),
            mask_channels=args.mask_channels,
            input_img_channels=args.input_img_channels,
            self_condition=args.self_condition
        )
        ## LOAD DATA ##
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False)
        if args.scale_lr:
            args.learning_rate = (
                    args.learning_rate * args.gradient_accumulation_steps * args.batch_size * device_num
            )
        ## Initialize optimizer
        if not args.use_lion:
            optimizer = AdamW(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        else:
            optimizer = Lion(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay
            )

        ## TRAIN MODEL ##
        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40,
                                                               verbose=True, min_lr=1e-4,cooldown=40)
        diffusion = MedSegDiff(
            model,
            timesteps=args.timesteps
        ).to(device)

        if args.load_model_from is not None:
            save_dict = torch.load(args.load_model_from)
            diffusion.load_state_dict(save_dict['model_state_dict'])
            optimizer.load_state_dict(save_dict['optimizer_state_dict'])
    else:
        local_rank=int(os.environ.get('LOCAL_RANK',-1))
        dist.init_process_group(backend='gloo')
        device=torch.device(f"cuda:{local_rank}")
        data_sampler=DistributedSampler(dataset)
        data_loader=torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size,
            sampler=data_sampler,
            shuffle=False,)
        model = Unet(
            dim=args.dim,
            image_size=args.image_size,
            dim_mults=(1, 2, 4, 8),
            mask_channels=args.mask_channels,
            input_img_channels=args.input_img_channels,
            self_condition=args.self_condition
        )
        if args.scale_lr:
            args.learning_rate = (
                    args.learning_rate * args.gradient_accumulation_steps * args.batch_size * device_num
            )
        ## Initialize optimizer
        if not args.use_lion:
            optimizer = AdamW(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        else:
            optimizer = Lion(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay
            )

        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40,
                                                               verbose=True, min_lr=1e-4,cooldown=40)

        diffusion=torch.nn.parallel.DistributedDataParallel(MedSegDiff(model,timesteps=args.timesteps).to(device),device_ids=[local_rank],output_device=local_rank)
        if args.load_model_from is not None:
            ###Note: if  the index of GPUs is not same , it would better to assign your pre-trained parameters to cpu then transfer them into GPU.
            save_dict = torch.load(args.load_model_from,map_location=torch.device('cpu'))['model_state_dict']
            if True:
                for k in list(save_dict.keys()):
                    newkey = k[7:]
                    save_dict[newkey] = save_dict.pop(k)
                diffusion.module.load_state_dict(save_dict)

            else:
                diffusion.model.load_state_dict(save_dict)
                optimizer.load_state_dict(save_dict['optimizer_state_dict'])






    import copy
    min_losses=1
    min_epoch=0
    for epoch in range(args.epochs):
        running_loss = 0.0
        iteration = 0
        loss_dict = {}
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        for (img, mask) in tqdm(data_loader):

            img=img.float()
            mask=mask.float()
            loss = diffusion(mask, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item() * img.size(0)
            loss_dict[str(iteration)] = '{:.8f}'.format(loss.item())
            iteration += 1

        counter += 1
        epoch_loss = running_loss/len(data_loader)


        scheduler.step(epoch_loss)
        Terminal_output='Training Loss : {:.6}, lr : {:.8f}'.format(epoch_loss,optimizer.state_dict()['param_groups'][0]['lr'])
        print(Terminal_output)

        if epoch % args.save_every == 0:
            if device_num ==1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))
            else:
                if dist.get_rank() ==0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': diffusion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))

        if epoch_loss < min_losses:

            former_best_name = os.path.join(checkpoint_dir, f'state_dict_best_epoch_{min_epoch}_loss_{min_losses}.pt')
            min_epoch = copy.deepcopy(epoch)
            min_losses = copy.deepcopy(epoch_loss)
            current_best_name = os.path.join(checkpoint_dir, f'state_dict_best_epoch_{epoch}_loss_{min_losses}.pt')
            if os.path.exists(former_best_name):
                os.remove(former_best_name)
            if device_num == 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, current_best_name)
            else:
                if dist.get_rank() == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': diffusion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, current_best_name)


if __name__ == '__main__':
    import time
    current_time = time.gmtime()
    str_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print ('Strating training at time {}'.format(str_time))
    main()

