# Copyright (C) 2024, Style-Splat
# All rights reserved.

# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting and Gaussian-Grouping
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import gc
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from argparse import ArgumentParser
from os import makedirs
from tqdm import tqdm
import open3d as o3d
import numpy as np
import torchvision
import shutil
import torch
import json
import functools
import os
import time
from gaussian_renderer.RenderDataset import RenderDataset
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from gs2gs.ip2p import InstructPix2Pix
from torch.utils.data import DataLoader

def cleanPointCloud(points, mask3d):
    mask3d = mask3d.bool().squeeze().cpu().numpy() # N,
    points = points.detach().cpu().numpy() # N x 3
    print("Before: ", np.sum(mask3d))
    object_points = points[mask3d]
    point_cloud = o3d.geometry.PointCloud()
    # EDIT: object_points was the wrong type for some reason
    object_points = np.array(object_points, dtype=np.float64)
    point_cloud.points = o3d.utility.Vector3dVector(object_points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=75, std_ratio=0.5)
    inlier_mask = np.zeros(object_points.shape[0], dtype=bool)
    inlier_mask[ind] = True
    updated_mask = mask3d.copy()
    updated_mask[mask3d] = inlier_mask
    print("After: ", np.sum(inlier_mask) )
    return updated_mask


class LossConfig:
    def __init__(self, 
                 device='cuda',
                 patch_size=32, 
                 use_lpips=True, 
                 lpips_loss_mult=3.0, 
                 interlevel_loss_mult=1.0, 
                 distortion_loss_mult=1.0, 
                 orientation_loss_mult=1.0, 
                 pred_normal_loss_mult=1.0, 
                 predict_normals=False):
        self.device = device
        self.patch_size = patch_size
        self.use_lpips = use_lpips
        self.lpips_loss_mult = lpips_loss_mult
        self.interlevel_loss_mult = interlevel_loss_mult
        self.distortion_loss_mult = distortion_loss_mult
        self.orientation_loss_mult = orientation_loss_mult
        self.pred_normal_loss_mult = pred_normal_loss_mult
        self.predict_normals = predict_normals

class LossComputation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
    def rgb_loss(self, target, prediction):
        # Implement the RGB loss (e.g., L2 loss or L1 loss)
        return torch.nn.functional.mse_loss(prediction, target)
    
    def l1_loss(self, target, prediction):
        return torch.nn.functional.l1_loss(prediction, target)   
    def lpips(self, pred_patches, gt_patches, device):
        lpips = LearnedPerceptualImagePatchSimilarity().to(device)
        return lpips(pred_patches, gt_patches)
    
    def get_loss_dict(self, output_images_rgb, images, device, metrics_dict=None):
        loss_dict = {}
        #images must be on right device
        loss_dict["l1_loss"] = self.l1_loss(images, output_images_rgb)
        loss_dict["rgb_loss"] = self.rgb_loss(images,output_images_rgb)
        if self.config.use_lpips:
            patch_size = self.config.patch_size
            out_patches = (
                output_images_rgb
                .view(-1, patch_size, patch_size, 3)
                .permute(0, 3, 1, 2) * 2 - 1
            )
            out_patches.clamp_(-1,1)
            gt_patches = (
                images
                .view(-1, patch_size, patch_size, 3)
                .permute(0, 3, 1, 2) * 2 - 1
            )
            gt_patches.clamp_(-1,1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches, device)
        return loss_dict    

def get_all_gt_images(viewpoint_stack, OBJ_ID, image_size):
    gt_images, all_mask2d = zip(*[(view.original_image, (view.objects == OBJ_ID).unsqueeze(0)) for view in viewpoint_stack])
    gt_images = torch.stack(gt_images)  # Shape: [N, C, H, W]
    all_mask2d = torch.cat(all_mask2d, dim=0).expand_as(gt_images)  # Shape: [N, 3, H, W]

    all_mask2d = all_mask2d.to(gt_images.device)
    gt_images = gt_images*all_mask2d
    gt_images = torch.nn.functional.interpolate(
        gt_images, size=(image_size, image_size), mode='bilinear', align_corners=False
    )  # Shape: [N, C, image_size, image_size]
    return gt_images
 

def get_memory_usage(return_all = True):
    # GPU memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
    if return_all:
        return f"GPU Allocated: {gpu_memory_allocated:.2f} GB | GPU Reserved: {gpu_memory_reserved:.2f} GB"
    else:
        return  f"GPU Allocated: {gpu_memory_allocated:.2f} GB"
    

def finetune_style(opt, model_path, views, gaussians, pipeline, background, classifier, OBJ_ID, epochs):

    loss_config = LossConfig()
    Loss = LossComputation(loss_config)
    #Configs 
    batch_size = 64  # Adjust based on available GPU memory
    edit_frequency = 20 #edit every x epochs
    
    guidance_scale: float = 12  
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.75
    """image guidance scale for InstructPix2Pix"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.7
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    image_size: int = 128
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[[OBJ_ID], :, :] > 0.95
        mask3d = mask.any(dim=0).squeeze()
        updated_mask = torch.Tensor(cleanPointCloud(gaussians._xyz, mask3d)).to(gaussians._xyz.device)
        mask3d = updated_mask[:,None,None]

    gaussians.finetune_setup(opt,mask3d)
    
    #PRE-PROCESS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ip2p = InstructPix2Pix(device, ip2p_use_full_precision=False)
 
    target_text = STYLE_TEXT
    target_text_features = ip2p.pipe._encode_prompt(
            target_text, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    target_text_features = torch.repeat_interleave(target_text_features,batch_size, dim=0)
    
    #Load GT Images
    gt_images = get_all_gt_images(views, OBJ_ID, image_size).to('cpu').detach()
    #Get 
    dataset = RenderDataset(views, gt_images, gaussians, pipeline, background,OBJ_ID, image_size,device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    losses = []
    mem_usage = get_memory_usage()
    total_batches = len(dataloader)
    progress_bar = tqdm(range(epochs), desc="Training:", dynamic_ncols=True)
    seen_indices = []
    for epoch, _ in enumerate(progress_bar): 
        # Initialize timing accumulators
        grab_time = 0
        edit_time = 0
        backprop_time = 0
        epoch_loss = 0
        progress_bar.set_postfix({
                "Batch": f"{1}/{total_batches}",
                "Avg Batch Loss": f"{0:.7f}",
                "Avg Epoch Loss": f"{sum(losses) / epoch if losses else 0 :.7f}",
                "Avg Grab Time (s)": f"{0:.4f}",
                "Avg Edit Time (s)": f"{0:.4f}",
                "Avg Loss Calc Time (s)": f"{0:.4f}",
                "Avg Backprop Time (s)": f"{0:.4f}",
                "GPU Mem": mem_usage,
            })
        
        for batch_idx, batch_data in enumerate(dataloader):
            
            # Time step 1: Grab Images from DataLoader
            start_time = time.time()
            batch_gt_images = batch_data["gt_images"].half().to(device)
            batch_rendered_images = batch_data["rendered_images"].squeeze(1).half().to(device)
            grab_time += time.time() - start_time
            indices = batch_data["indices"]
            
            # Time step 2: Editing the images
            start_time = time.time()         
            if epoch % edit_frequency == 0:
                with torch.no_grad():    
                    edited_batch = ip2p.edit_image(
                        target_text_features.to(device),
                        batch_rendered_images,
                        batch_gt_images,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        diffusion_steps=diffusion_steps,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                    dataset.update_edited_images(edited_batch, indices, seen_indices)
                    seen_indices.append(indices)
            else: 
                edited_batch  = []
                for i in indices:
                    val = dataset.previously_edited_images.get(int(i)).half()
                    edited_batch.append(val)
                edited_batch = torch.stack(edited_batch).to(device)
            edit_time += time.time() - start_time
            
            # Time step 3: Calculating the loss and doing backprop
            if edited_batch.shape[2:] != batch_rendered_images[0].shape[1:]:
                edited_batch = torch.nn.functional.interpolate(edited_batch, size=batch_rendered_images[0].shape[1:], mode='bilinear')

            loss_dict = Loss.get_loss_dict(batch_rendered_images, edited_batch, device)
            gaussians.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=False):
                loss = functools.reduce(torch.add, loss_dict.values())

            loss_scaler = 1e2
            loss = loss*loss_scaler
            epoch_loss+= loss.item()
            
            # List all tensors currently residing on GPU
            gc.collect()
            torch.cuda.empty_cache()   
            
            # Clear the gradients
            start_time = time.time()
            mem_usage = get_memory_usage() 
            loss.backward()
            backprop_time += time.time() - start_time 
            gaussians.optimizer.step()

            progress_bar.set_postfix({
                "Batch": f"{batch_idx+1}/{total_batches}",
                "Avg Batch Loss": f"{epoch_loss / (batch_idx+1):.7f}",
                "Avg Epoch Loss": f"{sum(losses) / epoch if losses else 0 :.7f}",
                "Avg Grab Time (s)": f"{grab_time / (batch_idx+1):.4f}",
                "Avg Loss Time (s)": f"{edit_time / (batch_idx+1):.4f}",
                "Avg Backprop Time (s)": f"{backprop_time / (batch_idx+1):.4f}",
                "GPU Mem": mem_usage,
            })
        losses.append(epoch_loss)
    
    progress_bar.close()

    
    point_cloud_path = os.path.join(model_path, f"point_cloud_style/{OBJ_ID}_{STYLE_TEXT_FILE}")
    makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print(F"point cloud path: {point_cloud_path}")
    
    # Plot the losses per iteration
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Save the plot in the graphs directory
    graphs_dir = "graphs"
    plot_path = os.path.join(graphs_dir, f"loss_plot_{OBJ_ID}_{STYLE_TEXT_FILE}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot at: {plot_path}")
    
    return gaussians, point_cloud_path

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    print("render")
    print(render_path)
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views[:30], desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        # Calculate and print the difference between the rendered image and the ground truth
        difference = torch.abs(rendering - gt)
        difference_path = os.path.join(render_path, '{0:05d}_diff.png'.format(idx))
        torchvision.utils.save_image(difference, difference_path)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def style(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : list,  epochs: int):
    # 1. load gaussian checkpoint
    for obj_id in select_obj_id:
        print("NOW DOING: " , STYLE_TEXT, obj_id)
        print()
        print()
        #determine SH_degree
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_classes = dataset.num_classes
        # print("Num classes: ",num_classes)
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"),weights_only=True))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. style selected object
        gaussians, pcd_path = finetune_style(opt, dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, classifier, obj_id, epochs)
        
        # 3. render new result
        dataset.object_path = 'object_mask'
        dataset.images = 'images'
        scene = Scene(dataset, gaussians, load_iteration=f'_style/{obj_id}_{STYLE_TEXT_FILE}', shuffle=False)
        with torch.no_grad():
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)
        print("PCD PATH::")
        print(pcd_path)
        shutil.rmtree(pcd_path)
            # if not skip_test:
            #     render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--style_text", default="", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true") 
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="", help="Path to the configuration file")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 256)
    args.select_obj_id = config.get("select_obj_id", [None])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.epochs = config.get("epoch", 20)
    STYLE_TEXT = args.style_text # "red"
    STYLE_TEXT_FILE = STYLE_TEXT.replace(" ", "_")
    safe_state(args.quiet)
    style(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.epochs)