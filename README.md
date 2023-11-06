# SegInpDiff - Denoising Diffusion Probabilistic Models for Simultaneous Echocardiography Image Segmentation and inpainting

___

---
## Repo architecture
med_seg_diff_pytorch.py contains the whole model architecture including the forward and reverse sampling process.

utils/dataset.py is used to process dataloader.

cross_att.py is the proposed cross attention module in this manuscript.

dice_loss.py is used to compute the dice scores.

driver.py-->training command includes the training process, as well as the forward process of transforming the original $x_0$ and its corresponding segmentation label into pure Gaussian noise.

sample.py-->sampling command to get inpainting $x_0$ and its corresponding segmentation prediction from pure Gaussian noise.

cond_mask_camus.png --> The large mask calculated from the CAMUS dataset. For a pixel, if the value of any segmentation result at the current pixel position is 1, the value of the current pixel is defined as 1. (logical OR operation)

plz see the details in the following introduction.

## Dataset preparation

The CAMUS echocardiography dataset is recommended !! 

If you do not want to rewrite the utils/dataset.py, please using the following dataset structure.

- CAMUS |  
  - training >  
    - patinet0001 >
		- img >
			- img0001.png, img0002.png.....
		- mask >
			- img0001.png, img0002.png.....
 - testing >  
	As same as the training.
			
The name of image and its correspongding mask should be same. 
			
	 


	 




## Training Commands
If you used single GPU, just run
```
python driver.py ####Note, DDPM takes a lot of resources, it's hard to run in single GPU.####

```
We have integerated nn.parallel.DistributedDataParallel training mode in driver.py, please use distributed.run mode,

e.g. --nproc_per_node=4 means 4Gpus, --multi_divice 0,1,2,3 means device index

```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 driver.py --multi_device 0,1,2,3 --batch_size 16
```

## Testing command
Note, you need to choose the specific epoch_number in fuction parse_args() in sample.py
```
python sample.py
```

