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
        #Render at the image size
        view.image_height = self.image_size
        view.image_width = self.image_size
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
        return {
            "indices": idx,
            "rendered_images": rendered_image,
            "gt_images": gt_image,
        }
    def update_edited_images(self, edited_batch, indices, expected_indices):
        #self.previously_edited_images.update({idx: img.cpu() for idx, img in zip(indices, edited_batch)})
        for idx, img in zip(indices, edited_batch):
            self.previously_edited_images[int(idx)] = img.cpu() 