from torch.utils.data import Dataset
from gaussian_renderer import render
import torch

class RenderDataset(Dataset):
    def __init__(self, viewpoints, gt_images, gaussians, pipeline, background,obj_id, image_size, device):
        self.viewpoints = viewpoints  # Dynamic rendering inputs
        self.gt_images = gt_images    # Static preloaded GT images
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.obj_id = obj_id 
        self.image_size = image_size
        self.device = device 
        self.previously_edited_images = {}

    def __len__(self):
        return len(self.viewpoints)

    def __getitem__(self, idx):
        # Load GT image (static)
        gt_image = self.gt_images[idx]
        # Dynamically render the image
        view = self.viewpoints[idx]
        view.image_height = 224
        view.image_width = 224
        render_pkg = render(view, self.gaussians, self.pipeline, self.background)
        rendered_image = render_pkg["render"].to("cpu")
        mask2d = view.objects == self.obj_id
        mask2d = mask2d.to(torch.float16)    # Convert to float32
        mask2d = mask2d.unsqueeze(0).to(rendered_image.device)
        mask2d = torch.nn.functional.interpolate(
            mask2d, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )
        mask2d = (mask2d >= 0.5).to(torch.bool)
        # Apply masks and resizing
        rendered_image = rendered_image * mask2d
        #rendered_image = torch.nn.functional.interpolate(
        #    rendered_image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        #)
        return {
            "indices": idx,
            "rendered_images": rendered_image,
            "gt_images": gt_image,
        }