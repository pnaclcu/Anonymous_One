import copy
import os
import argparse
import torchvision
import cv2
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms
from med_seg_diff_pytorch.med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
#from accelerate import Accelerator
import skimage.io as io
from utils.dataset import BasicDataset
import random
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte
from dice_loss import dice_coeff
import numpy as np
torch.backends.cudnn.enabled = False
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


## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-img', '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=3,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=112, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='./data', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='CAMUS', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=20, help='save_every n epochs (default: 100)')
    parser.add_argument('--num_ens', type=int, default=1,
                        help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
    parser.add_argument('--load_model_from', default='./output/checkpoints_camus/', help='path to pt file to load from')
    parser.add_argument('--multi_gpus', default=True, help='path to pt file to load from')
    parser.add_argument('--save_uncertainty', action='store_true',default=True,
                        help='Whether to store the uncertainty in predictions (only works for ensablmes)')
    parser.add_argument('--epoch_number',default='best',
                        help='Test the model epoch number')
    parser.add_argument('--cuda_device',default=0,help='cuda device number')

    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=False,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=True)
    elif args.dataset =='RV':
        args.data_path='../RV_data_parts'
        args.load_model_from='./output/checkpoints_RV/'

        transform_list=[transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        val_percent=0.2
        all_patient = acquire_all_patient(dir=args.data_path)

        train_patient, test_patient, _, _ = train_test_split(all_patient, all_patient, test_size=val_percent,
                                                             random_state=42)

        test_img, test_mask = acquire_all_img(test_patient)
        dataset = BasicDataset(test_img, test_mask, transform=transform_train,scale=1 if args.image_size==256 else 0.5,training=False)

    elif  args.dataset =='CAMUS':
        args.data_path='../CAMUS'
        args.load_model_from = './output/checkpoints_camus/'
        args.image_size = 112
        transform_list=[transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        train_path=os.path.join(args.data_path,'training')
        test_path=os.path.join(args.data_path,'testing')
        train_patient=os.listdir(train_path)
        test_patient=os.listdir(test_path)
        train_patient=[os.path.join(train_path,i) for i in train_patient]
        test_patient = [os.path.join(test_path, i) for i in test_patient]

        test_img, test_mask = acquire_all_img(test_patient)
        dataset = BasicDataset(test_img, test_mask, transform=transform_train,scale=1 if args.image_size==256 else 0.5,training=False)


    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    return training_generator







def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.dataset=='CAMUS':
        inference_dir = os.path.join(args.output_dir, 'inference_camus')

    else:
        inference_dir = os.path.join(args.output_dir, 'inference_RV')

    os.makedirs(inference_dir, exist_ok=True)


    ## DEFINE MODEL ##
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=(1, 2, 4, 8),
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition
    )

    ## LOAD DATA ##
    data_loader = load_data(args)

    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).to('cuda:{}'.format(args.cuda_device))#.to(accelerator.device)
    ###Load the large mask to cover the whole cardiac chamber. The corresponding mask is computed from the whole datasetk###
    ###Perform an OR operation on the same pixel across all images, meaning that if any pixel is 1, it is defined as 1###
    if args.dataset=='CAMUS':

        cond_mask = Image.open('cond_mask_camus.png').convert('L')
    else:

        cond_mask=Image.open('cond_mask_RV.png').convert('L')



    cond_mask = cond_mask.resize((args.image_size, args.image_size))
    cond_mask=torch.from_numpy(np.array(cond_mask))
    cond_mask = cond_mask.float() / 255
    cond_mask = (cond_mask > 0.0).float()

    import time

    if os.path.isdir(args.load_model_from):
        all_model=os.listdir(args.load_model_from)
        for model_name in all_model:
            if 'best' not in model_name:
                continue
            model_path=os.path.join(args.load_model_from,model_name)

            modification_time = os.path.getmtime(model_path)
            local_time = time.localtime(modification_time)
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
            print(model_name, formatted_time)
            current_epoch_number=args.epoch_number

            save_dict=torch.load(model_path,map_location='cuda:{}'.format(args.cuda_device))['model_state_dict']
            if args.multi_gpus:
                for k in list(save_dict.keys()):
                    newkey = k[7:]
                    save_dict[newkey] = save_dict.pop(k)
                diffusion.load_state_dict(save_dict)
            else:
                diffusion.load_state_dict(save_dict)

            for (imgs, masks, fnames) in tqdm(data_loader):
                # pre allocate preds
                imgs = imgs.float()
                masks = masks.float()

                pred_masks = torch.zeros((imgs.shape[0], args.num_ens, imgs.shape[2], imgs.shape[3]))
                pred_c1 = torch.zeros((imgs.shape[0], args.num_ens, imgs.shape[2], imgs.shape[3]))
                pred_c2 = torch.zeros((imgs.shape[0], args.num_ens, imgs.shape[2], imgs.shape[3]))
                pred_c3 = torch.zeros((imgs.shape[0], args.num_ens, imgs.shape[2], imgs.shape[3]))

                for i in range(args.num_ens):
                    img_c_mask=diffusion.sample(imgs,cond_mask).cpu().detach()
                    sampled_mask=img_c_mask[:, 0, :, :].unsqueeze(1)
                    sampled_c1 = img_c_mask[:, 1, :, :].unsqueeze(1)
                    sampled_c2 = img_c_mask[:, 2, :, :].unsqueeze(1)
                    sampled_c3 = img_c_mask[:, 3, :, :].unsqueeze(1)

                    pred_masks[:,i:i+1,:,:]=sampled_mask
                    pred_c1[:, i:i + 1, :, :]=sampled_c1
                    pred_c2[:, i:i + 1, :, :]=sampled_c2
                    pred_c3[:, i:i + 1, :, :]=sampled_c3



                preds_mask_mean = pred_masks.mean(dim=1)
                pred_c1_mean = pred_c1.mean(dim=1).unsqueeze(1)
                pred_c2_mean = pred_c2.mean(dim=1).unsqueeze(1)
                pred_c3_mean = pred_c3.mean(dim=1).unsqueeze(1)
                pred_imgs_mean=torch.cat((pred_c1_mean,pred_c2_mean,pred_c3_mean),dim=1)



                for idx in range(preds_mask_mean.shape[0]):
                    abs_path = fnames[idx].split('/')
                    frame_num = abs_path[-1]
                    out_name_img = copy.deepcopy(frame_num)

                    if 'png' in frame_num:
                        frame_num=frame_num.split('.png')[0]
                    patient_num = abs_path[-3]
                    group_num = abs_path[-4]
                    temp_list=[]

                    #default binary threshold=0.5 if using num_ens=1.
                    #if using num_ens != 1, there is no need to USE default binary threshold 0.5

                    current_dice = dice_coeff((preds_mask_mean[idx, :, :] > 0.5).float(),masks[idx, :, :].squeeze()).item()
                    #current_dice = dice_coeff((preds_mask_mean[idx, :, :]).float(),masks[idx, :, :].squeeze()).item()

                    if args.dataset == 'RV':
                        out_name_mask = '{}_pred_dice_{}.png'.format(patient_num+'_'+frame_num,round(current_dice,3))
                    elif args.dataset =='CAMUS':
                        out_name_mask='{}_pred_dice_{}.png'.format(frame_num,round(current_dice,3))

                    base_path=os.path.join(inference_dir, model_name,patient_num)
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    mask_mean_name=os.path.join(base_path,out_name_mask)

                    io.imsave(mask_mean_name, img_as_ubyte(preds_mask_mean[idx, :, :] > 0.5))

                    if args.dataset == 'CAMUS':
                        img_mean_name = os.path.join(base_path, out_name_img)
                    elif args.dataset == 'RV':
                        img_mean_name = os.path.join(base_path, patient_num+'_'+out_name_img)

                    io.imsave(img_mean_name,img_as_ubyte(pred_imgs_mean[idx, :, :]).transpose(1,2,0))






if __name__ == '__main__':
    main()
