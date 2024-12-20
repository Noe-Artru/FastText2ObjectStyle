# Copyright (C) 2024, Style-Splat
# All rights reserved.

# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting, Gaussian-Grouping and Style-Splat
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import clip
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import torch
import gc
from argparse import ArgumentParser
from random import randint
from os import makedirs
from tqdm import tqdm
import open3d as o3d
import numpy as np
import torchvision
import shutil
import torch
import json
import os
from torch.utils.data import DataLoader
from torchvision import models
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from gaussian_renderer.RenderDataset import RenderDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

def random_words(n): # Selects Random words as negative prompts
    with open("wordlist.10000.txt", 'r') as file:
        words = file.read().split()

    return random.sample(words, n)

def cleanPointCloud(points, mask3d): # Filters out outliers from the point cloud
    mask3d = mask3d.bool().squeeze().cpu().numpy() # N,
    points = points.detach().cpu().numpy() # N x 3
    print("Before: ", np.sum(mask3d))
    object_points = points[mask3d]
    point_cloud = o3d.geometry.PointCloud()
    object_points = np.array(object_points, dtype=np.float64)
    point_cloud.points = o3d.utility.Vector3dVector(object_points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=75, std_ratio=0.5)
    inlier_mask = np.zeros(object_points.shape[0], dtype=bool)
    inlier_mask[ind] = True
    updated_mask = mask3d.copy()
    updated_mask[mask3d] = inlier_mask
    print("After: ", np.sum(inlier_mask) )
    return updated_mask


def regularization_loss(original_images, modified_images, model):
    features_original = model(original_images)
    features_modified = model(modified_images)
    loss = 0
    for f1, f2 in zip(features_original, features_modified):
        loss += F.mse_loss(f1, f2)
    return loss

def get_all_gt_images(viewpoint_stack, OBJ_ID, image_size):
    gt_images, all_mask2d = zip(*[(view.original_image, (view.objects == OBJ_ID).unsqueeze(0)) for view in viewpoint_stack])
    gt_images = torch.stack(gt_images)  # Shape: [N, C, H, W]
    all_mask2d = torch.cat(all_mask2d, dim=0).expand_as(gt_images)  # Shape: [N, 3, H, W]

    all_mask2d = all_mask2d.to(gt_images.device)
    gt_images = gt_images*all_mask2d
    gt_images = torch.nn.functional.interpolate(
    gt_images, size=(image_size, image_size), mode='bilinear', align_corners=False)
    return gt_images

def get_memory_usage(return_all = True):
    gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
    if return_all:
        return f"GPU Allocated: {gpu_memory_allocated:.2f} GB | GPU Reserved: {gpu_memory_reserved:.2f} GB"
    else:
        return  f"GPU Allocated: {gpu_memory_allocated:.2f} GB"
 

def finetune_style(opt, model_path, views, gaussians, pipeline, background, classifier, OBJ_ID, epoch, scale_reg_loss):
    batch_size = 16 # 32 For roughly 30GB RAM
    image_size = 224
    epochs = epoch
    
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[[OBJ_ID], :, :] > 0.95
        mask3d = mask.any(dim=0).squeeze()
        updated_mask = torch.Tensor(cleanPointCloud(gaussians._xyz, mask3d)).to(gaussians._xyz.device)
        mask3d = updated_mask[:,None,None]

    gaussians.finetune_setup(opt,mask3d)


    #CLIP preprocessing
    target_text = STYLE_TEXT
    neg = random_words(10)
    print("Chosen negative words: ", neg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    target_text_features = model.encode_text(clip.tokenize(target_text).to(device))
    target_neg_text_features = model.encode_text(clip.tokenize(neg).to(device))
    target_text_features_normed = target_text_features / target_text_features.norm(dim=-1, keepdim=True)
    target_neg_text_features_normed = target_neg_text_features / target_neg_text_features.norm(dim=-1, keepdim=True)
    
    clip_preprocess = T.Compose([
        T.Normalize(                    # Normalize using CLIP's mean and std
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
    ])

    # VGG16 for regularization loss
    vgg = models.vgg16(pretrained=True).features[:16].to(device).half()
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False

    #Load GT Images
    gt_images = get_all_gt_images(views, OBJ_ID, image_size).half().to('cpu').detach()
     
    dataset = RenderDataset(views, gt_images, gaussians, pipeline, background,OBJ_ID, image_size,device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    losses = []
    mem_usage = get_memory_usage()
    total_batches = len(dataloader)
    progress_bar = tqdm(range(epochs), desc="Training:", dynamic_ncols=True)
    gc.collect()
    torch.cuda.empty_cache()   
    for epoch, _ in enumerate(progress_bar): 
        # Initialize timing accumulators
        grab_time = 0
        loss_time = 0
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
        for batch_idx, batch_data in enumerate(dataloader): # Main Training Loop
            
            #Render batch of current scene
            start_time = time.time()
            batch_rendered_images = batch_data["rendered_images"].squeeze(1).half().to(device)
            preprocessed_imgs = clip_preprocess(batch_rendered_images).to(device)
            image_encoding = model.encode_image(preprocessed_imgs)
            image_encoding_normed = image_encoding/image_encoding.norm(dim=-1, keepdim=True)
            grab_time += time.time()-start_time

            # Compute cosine similarities
            start_time = time.time()  
            similarity = torch.nn.functional.cosine_similarity(image_encoding_normed, target_text_features_normed.detach())
            neg_similarity = torch.nn.functional.cosine_similarity(image_encoding_normed.unsqueeze(1), target_neg_text_features_normed.unsqueeze(0).detach(), dim=-1)

            # Compute loss
            temperature = 1
            loss = -torch.log(torch.exp(similarity/temperature) / (torch.exp(similarity/temperature) + torch.sum(torch.exp(neg_similarity/temperature)))).sum()
            regularization_loss = regularization_loss(batch_data["gt_images"].to(device).half(), batch_rendered_images, vgg)
            loss = loss + regularization_loss * scale_reg_loss
            epoch_loss += loss.item()
            loss_time+= time.time() - start_time
            gaussians.optimizer.zero_grad(set_to_none=True)

            start_time = time.time()
            mem_usage = get_memory_usage() 
            loss.backward()
            gaussians.optimizer.step()
            backprop_time += time.time() - start_time 


            progress_bar.set_postfix({
                "Batch": f"{batch_idx+1}/{total_batches}",
                "Avg Batch Loss": f"{epoch_loss / (batch_idx+1):.7f}",
                "Avg Epoch Loss": f"{sum(losses) / epoch if losses else 0 :.7f}",
                "Avg Grab Time (s)": f"{grab_time / (batch_idx+1):.4f}",
                "Avg Loss Time (s)": f"{loss_time / (batch_idx+1):.4f}",
                "Avg Backprop Time (s)": f"{backprop_time / (batch_idx+1):.4f}",
                "GPU Mem": mem_usage,
            })
        losses.append(epoch_loss)
        gc.collect()
        torch.cuda.empty_cache() 
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
    plot_path = os.path.join(graphs_dir, f"loss_plot_{OBJ_ID}_{STYLE_TEXT}.png")
    i = 0
    new_plot_path = plot_path
    while os.path.exists(new_plot_path):
        new_plot_path = f"{plot_path.rsplit('.', 1)[0]}_{i}.png"
        i += 1
    plt.savefig(new_plot_path)
    plt.close()
    print(f"Saved loss plot at: {new_plot_path}")
    
    return gaussians, point_cloud_path

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    new_render_path = render_path
    i = 0
    while os.path.exists(new_render_path):
        new_render_path = f"{render_path.rsplit('.', 1)[0]}_{i}"
        i += 1

    makedirs(new_render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views[:30], desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        torchvision.utils.save_image(rendering, os.path.join(new_render_path, '{0:05d}'.format(idx) + ".png"))


def style(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : list, removal_thresh : float,  epoch: int, scale_reg_loss: float):
    # 1. load Gaussian checkpoint
    for obj_id in select_obj_id:
        print("NOW DOING: " , STYLE_TEXT, obj_id)
        print()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_classes = dataset.num_classes
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. style selected object (Main Training Loop)
        gaussians, pcd_path = finetune_style(opt, dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, classifier, obj_id, epoch, scale_reg_loss)
        
        # 3. Render stylized result
        dataset.object_path = 'object_mask'
        dataset.images = 'images'
        scene = Scene(dataset, gaussians, load_iteration=f'_style/{obj_id}_{STYLE_TEXT_FILE}', shuffle=False)
        with torch.no_grad():
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)
        print("PCD PATH::")
        print(pcd_path)
        shutil.rmtree(pcd_path) #Avoids accumulating point clouds
        

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
    parser.add_argument("--scale_reg_loss", default=0, type=float)
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
    args.epoch = config.get("epoch", 2000)

    # Set the style text
    STYLE_TEXT = args.style_text
    STYLE_TEXT_FILE = STYLE_TEXT.replace(" ", "_")
    safe_state(args.quiet)

    # Run style transfer
    style(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.epoch, args.scale_reg_loss)