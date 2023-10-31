# SegInpDiff - Denoising Diffusion Probabilistic Models for Simultaneous Echocardiography Image Segmentation and inpainting

___

---

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

