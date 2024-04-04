import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
from scipy.spatial.transform import Rotation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = []
    if incl_input:
        results.append(x)
        
    # encode input tensor and append the encoded tensor to the list of results.
    freq = torch.pow(2,torch.arange(num_frequencies)).to(device)
    sin_list = torch.sin(torch.pi*freq*x.reshape((-1,1))).to(device)
    cos_list  = torch.cos(torch.pi*freq*x.reshape((-1,1))).to(device)
    results.append(sin_list.reshape((x.shape[0],-1)))
    results.append(cos_list.reshape((x.shape[0],-1)))
    
    return torch.cat(results, dim=-1)

# Load input images, poses, and intrinsics
data = np.load("lego_data.npz")

# Images
images = data["images"]

# Height and width of each image
height, width = images.shape[1:3]

# Camera extrinsics (poses)
poses = data["poses"]
poses = torch.from_numpy(poses).to(device)

# Camera intrinsics
intrinsics = data["intrinsics"]
intrinsics = torch.from_numpy(intrinsics).to(device)

# Hold one image out (for test).
test_image, test_pose = images[101], poses[101]
test_image = torch.from_numpy(test_image).to(device)

print(test_pose)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

# plt.imshow(test_image.detach().cpu().numpy())
# plt.show()

def stratified_sampling(ray_origins, ray_directions, near, far, samples):


 
    i = torch.arange(samples)
    depth = near + (i/samples)*(far-near)
    depth = depth.to(device)
    ray_points = torch.unsqueeze(ray_origins, dim=3) + torch.unsqueeze(ray_directions, dim = 3) * depth
    ray_points = ray_points.permute(0,1,3,2)
    depth_points = torch.broadcast_to(depth, (ray_origins.shape[0], ray_origins.shape[1], samples))
   
    return ray_points, depth_points

def get_rays(height, width, intrinsics, w_R_c, w_T_c):

   
    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

   
    coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy'), -1).reshape((-1,2))
    coords = torch.concat((coords, torch.ones(coords.shape[0],1)), -1).to(device)
    # print(w_R_c.dtype, intrinsics.dtype, coords.dtype, w_T_c.dtype)
    rays = w_R_c @ torch.linalg.inv(intrinsics) @ coords.T
    # print(rays[:2,...])
    # print(rays.shape)
    rays = rays/torch.norm(rays, dim=0)
    rays_directions = rays.T.reshape((height,width,3))
    rays_origins = torch.broadcast_to(w_T_c.T, ((height, width,3)))
    # print(rays_directions.shape, rays_directions[0,:2,:])
    # print(rays_origins.shape, rays_origins[0,:2,:])
 
    return rays_origins, rays_directions

def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

  
    coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy'), -1).reshape((-1,2))
    coords = torch.concat((coords, torch.ones(coords.shape[0],1)), -1).to(device)
    # print(w_R_c.dtype, intrinsics.dtype, coords.dtype, w_T_c.dtype)
    rays = w_R_c @ torch.linalg.inv(intrinsics) @ coords.T
    # print(rays[:2,...])
    # print(rays.shape)
    rays = rays/torch.norm(rays, dim=0)
    rays_directions = rays.T.reshape((height,width,3))
    rays_origins = torch.broadcast_to(w_T_c.T, ((height, width,3)))
    # print(rays_directions.shape, rays_directions[0,:2,:])
    # print(rays_origins.shape, rays_origins[0,:2,:])
  
    return rays_origins, rays_directions

class nerf_model(nn.Module):

    
    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

       
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(3*2*num_x_frequencies + 3, filter_size),
            'layer_2': nn.Linear(filter_size,filter_size),
            'layer_3': nn.Linear(filter_size,filter_size),
            'layer_4': nn.Linear(filter_size,filter_size),
            'layer_5': nn.Linear(filter_size,filter_size),
            'layer_6': nn.Linear(filter_size+3*2*num_x_frequencies + 3, filter_size),
            'layer_7': nn.Linear(filter_size,filter_size),
            'layer_8': nn.Linear(filter_size,filter_size),
            'layer_s': nn.Linear(filter_size,1),
            'layer_9': nn.Linear(filter_size,filter_size),
            'layer_10': nn.Linear(filter_size+3*2*num_d_frequencies + 3,filter_size//2),
            'layer_11': nn.Linear(filter_size//2, 3),
        })

      

    def forward(self, x, d):
       
        out = self.layers['layer_1'](x)
        out = F.relu(out)
        out = self.layers['layer_2'](out)
        out = F.relu(out)
        out = self.layers['layer_3'](out)
        out = F.relu(out)
        out = self.layers['layer_4'](out)
        out = F.relu(out)
        out = self.layers['layer_5'](out)
        out = F.relu(out)
        out = torch.concat([x,out], -1)
        out = self.layers['layer_6'](out)
        out = F.relu(out)
        out = self.layers['layer_7'](out)
        out = F.relu(out)
        out = self.layers['layer_8'](out)
        sigma = self.layers['layer_s'](out)
        sigma = F.relu(sigma)
        out = self.layers['layer_9'](out)
        out = F.relu(out)
        out = torch.concat([out, d], -1)
        out = self.layers['layer_10'](out)
        out = F.relu(out)
        out = self.layers['layer_11'](out)
        rgb = F.sigmoid(out)

        return rgb, sigma
    
def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies, pbp_weights=None):
    def get_chunks(inputs, chunksize = 2**15):
        
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    ray_directions = torch.unsqueeze(ray_directions, dim=2)
    ray_directions = torch.broadcast_to(ray_directions, ray_points.shape)
    # print('ray_direction shape:', ray_directions.shape)
    # print('ray points shape:', ray_points.shape)
    embed_ray_dir = positional_encoding(ray_directions.reshape((-1,3)), num_frequencies=num_d_frequencies, incl_input=True)
    embed_ray_point = positional_encoding(ray_points.reshape((-1,3)), num_frequencies=num_x_frequencies, incl_input=True)
    if pbp_weights is not None:
        # embeddings are 
        # pbp_weights should be (L,)
        embed_ray_dir *= pbp_weights[None, None, :]
    ray_points_batches = get_chunks(embed_ray_point)
    ray_directions_batches = get_chunks(embed_ray_dir)
    # ray_points_batches = torch.concat(ray_points_batches, 0)
    # ray_directions_batches = torch.concat(ray_directions_batches, 0)
    # print('ray_direction batches shape:', ray_directions_batches.shape)
    # print('ray points batches shape:', ray_points_batches.shape)


    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

   
    delta = depth_points[...,1:]  - depth_points[...,:-1]
    delta = torch.concat([delta, torch.ones((delta.shape[0], delta.shape[1], 1)).to(device)*1e9], -1) #
    inter_Ti = torch.exp(-s*delta)
    Ti = torch.concat([torch.ones((delta.shape[0], delta.shape[1], 1)).to(device),torch.cumprod(inter_Ti, dim=-1)[:,:,:-1]], -1)

    c = torch.unsqueeze(Ti, dim=-1) * (1-torch.unsqueeze(inter_Ti, dim=-1)) * rgb
    rec_image = torch.sum(c, dim=2)

 

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies, pbp_weights=None):

    
    w_R_c = pose[:3,:3]
    w_T_c = pose[:3,3]
    ray_o, ray_d = get_rays(height, width, intrinsics, w_R_c.reshape((3,3)), w_T_c.reshape((3,1)))

    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_o, ray_d, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_d, num_x_frequencies, num_d_frequencies, pbp_weights=pbp_weights)
    rgb_list = []
    sigma_list =[]
    #forward pass the batches and concatenate the outputs at the end
    for batch_i in range(len(ray_points_batches)):
        #forward pass the batches and concatenate the outputs at the end
        rgb, sigma = model(ray_points_batches[batch_i], ray_directions_batches[batch_i])
        rgb_list.append(rgb)
        sigma_list.append(sigma)
        # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(torch.concat(rgb_list, 0).reshape((height, width,samples, 3)), torch.concat(sigma_list,0).reshape((height, width, samples)), depth_points)
  
    return rec_image

num_x_frequencies = 10
num_d_frequencies = 4
learning_rate  = 5e-4
iterations = 3000
samples = 100
display = 25
near = 0.667
far = 2

model = nerf_model(num_x_frequencies=num_x_frequencies,num_d_frequencies=num_d_frequencies).to(device)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

psnrs = []
iternums = []

t = time.time()
t0 = time.time()

criterion = torch.nn.MSELoss()

class BARFStyle(nn.Module):
    def __init__(self, nerf, initial_guess, intrinsics, near, far, Lx=10, Ld=4, tag=None):
        super().__init__()
        
        self.intrinsics = intrinsics
        self.near = near
        self.far = far
        self.Lx = Lx
        self.Ld = Ld
        
        self.tag = tag
        
        
        self.initial_guess = initial_guess.clone()
        self.nerf = nerf.to(device)
        self.pose_param = nn.Parameter(initial_guess.clone())
        
        self._recompute_pose_intermediary()
        
        self.nerf_opt = torch.optim.SGD(self.nerf.parameters(), lr=1e-2)
        self.pose_opt = torch.optim.SGD([self.pose_param], lr=1e-2)
        
        self.pbp_weights_x = None
        self.pbp_weights_d = None
        self.criterion = nn.MSELoss()
        
        self.alpha = 0
    
    def _recompute_pose_intermediary(self):
        rot_x = torch.stack([
            torch.tensor([1.0, 0.0 ,0.0], device=device),
            torch.cat([torch.zeros((1,), device=device), self.pose_param[3:4].cos(), -self.pose_param[3:4].sin()]),
            torch.cat([torch.zeros((1,), device=device), self.pose_param[3:4].sin(), self.pose_param[3:4].cos()]),
        ], dim=0)
        rot_y = torch.stack([
            torch.cat([self.pose_param[4:5].cos(), torch.zeros((1,), device=device), self.pose_param[4:5].sin()]),
            torch.tensor([0.0, 1.0, 0.0], device=device),
            torch.cat([-self.pose_param[4:5].sin(), torch.zeros((1,), device=device), self.pose_param[4:5].cos()]),
        ], dim=0)
        rot_z = torch.stack([
            torch.cat([self.pose_param[5:6].cos(), -self.pose_param[5:6].sin(), torch.zeros((1,), device=device)]),
            torch.cat([self.pose_param[5:6].sin(), self.pose_param[5:6].cos(), torch.zeros((1,), device=device)]),
            torch.tensor([0.0, 0.0, 1.0], device=device),
        ], dim=0)
        rot_xyz = rot_z @ rot_y @ rot_x
        
        self.pose_follow = torch.cat([
            torch.cat([rot_xyz, self.pose_param[:3].reshape(3, 1)], dim=-1),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).reshape(1, 4),
        ], dim=0)
    
    def compute_pbp(self, iter_num):
        fxs = torch.arange(0, self.Lx)
        fds = torch.arange(0, self.Ld)
        
        
        
    
    def fb_pass(self, target_image, iter_num):
        self.nerf_opt.zero_grad()
        self.pose_opt.zero_grad()
        self._recompute_pose_intermediary()
        
        height, width = target_image.shape[:2]
        rec_image = one_forward_pass(height, width, self.intrinsics, self.pose_follow,
                                     self.near, self.far, 64, self.nerf,
                                     self.Lx, self.Ld, pbp_weights=None)
        
        loss = self.criterion(rec_image, target_image)
        loss.backward()
        
        self.pose_opt.step()
        # self.nerf_opt.step()
        
        
        if iter_num % 30 == 0:
            plt.subplots(1, 2)
            plt.subplot(1, 2, 1)
            plt.imshow(rec_image.cpu().detach().numpy())
            plt.title(f'Reconstructed at {iter_num}')
            
            plt.subplot(1, 2, 2)
            plt.imshow(target_image.cpu().detach().numpy())
            plt.title('Target')
            
            plt.savefig(f'vis/barf_style{"_"+self.tag if self.tag is not None else ""}_{iter_num}.png')
        
        return loss.item()

def main(test_type, tag=None):
    if test_type == 0:
        
        global test_pose, test_image, intrinsics, near, far
        # lego nerf
        core_nerf = nerf_model(num_x_frequencies=10, num_d_frequencies=4)
        core_nerf.load_state_dict(torch.load('model_nerf_alt.pt'))

        # first attempt at random rotations
        # randu, _, randv = (0.01 * torch.randn(3, 3, device=device)).svd()
        # rand_rot = randu @ randv.T
        # rand_rot *= rand_rot.det().sign()
        # new_rot = test_pose[:3, :3] @ rand_rot 

        # assert torch.allclose(rand_rot @ rand_rot.T, torch.eye(3, device=device), atol=1e-5), 'Orthogonality violation'
        # assert torch.allclose(rand_rot.det(), torch.ones([1,], device=device), atol=1e-5), 'Unit determinant violation'

        # second attempt
        pose_noise = 0.1 # meters
        angle_noise = 0.01 # radians
        r =  Rotation.from_euler('xyz', np.random.randn(3)*angle_noise, degrees=False)
        rot_d = torch.from_numpy(r.as_matrix()).float().to(device)

        new_pose = torch.eye(4, device=device)
        # new_pose[:3, :3] = new_rot
        new_pose[:3, :3] = rot_d @ test_pose[:3, :3]
        new_pose[:3, 3] = test_pose[:3, 3] + torch.randn(3, device=device) * pose_noise
        
        r =  Rotation.from_matrix(new_pose[:3, :3].detach().cpu().numpy())
        angles = r.as_euler("xyz", degrees=False)
        # print(angles)

        # barf = BARFStyle(core_nerf, new_pose, intrinsics=intrinsics, near=near, far=far)
        pose_init = torch.cat([new_pose[:3, 3], torch.from_numpy(angles).to(device)]).float()
        barf = BARFStyle(core_nerf, pose_init, intrinsics=intrinsics, near=near, far=far)
        barf.to(device)
        
        with torch.autograd.set_detect_anomaly(True):
            for i in range(121):
                loss = barf.fb_pass(test_image, iter_num=i)
                print(f'Step {i}: {loss}')
        
        print(f'Initial pose:\n{new_pose}')
        print(f'Ground truth pose:\n{test_pose}')
        print(f'Final estimate:\n{barf.pose_param}')
    
    else:
        from odom_experiment import (read_img_to_tensor, read_matrices)
        # icl-nuim nerf
        
        core_nerf = nerf_model(num_x_frequencies=15, num_d_frequencies=6)
        core_nerf.load_state_dict(torch.load(f'icl_nuim_model_nerf{"_"+tag if tag is not None else ""}.pt'))
        
        test_idx = 100
        
        all_poses = read_matrices('livingRoom2n.gt.sim')
        test_pose = all_poses[test_idx].to(device)
        
        test_image = read_img_to_tensor(test_idx, 100, 100).to(device)
        
        pose_noise = 0.5 # meters
        angle_noise = np.deg2rad(10) # radians
        r =  Rotation.from_euler('xyz', np.random.randn(3)*angle_noise, degrees=False)
        rot_d = torch.from_numpy(r.as_matrix()).float().to(device)

        new_pose = torch.eye(4, device=device)
        # new_pose[:3, :3] = new_rot
        new_pose[:3, :3] = rot_d @ test_pose[:3, :3]
        new_pose[:3, 3] = test_pose[:3, 3] + torch.randn(3, device=device) * pose_noise
        
        r =  Rotation.from_matrix(new_pose[:3, :3].detach().cpu().numpy())
        angles = r.as_euler("xyz", degrees=False)
        # print(angles)

        # barf = BARFStyle(core_nerf, new_pose, intrinsics=intrinsics, near=near, far=far)
        near = 2
        far = 6
        intrinsics = torch.tensor([
            [481.20/640*100, 0, 49.50],
            [0, -480.00/480*100, 49.50],
            [0, 0, 1],
        ]).to(device)
        pose_init = torch.cat([new_pose[:3, 3], torch.from_numpy(angles).to(device)]).float()
        barf = BARFStyle(core_nerf, pose_init, intrinsics=intrinsics, near=near, far=far, Lx=15, Ld=6, tag=tag)
        barf.to(device)
        
        with torch.autograd.set_detect_anomaly(True):
            for i in range(121):
                loss = barf.fb_pass(test_image, iter_num=i)
                print(f'Step {i}: {loss}')
        
        print(f'Initial pose:\n{new_pose}')
        print(f'Ground truth pose:\n{test_pose}')
        print(f'Final estimate:\n{barf.pose_param}')

if __name__ == '__main__':
    test_type = 1
    tag = 'v2'
    main(test_type=test_type, tag=tag)