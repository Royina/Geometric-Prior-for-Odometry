import torch
import numpy as np
import time
from barf_style import (
    nerf_model,
    one_forward_pass,
)
from matplotlib import pyplot as plt
import cv2
import os

vis_path = './vis'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def read_matrices(file_path):
    matrices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(0, len(lines), 4):
            matrix_lines = lines[i:i+4]
            matrix = torch.tensor([
                list(map(float, line.split())) if len(line) > 1 \
                    else [0.0, 0.0, 0.0, 1.0] for line in matrix_lines
            ])
            matrices.append(matrix)

    return torch.stack(matrices, dim=0)

all_poses = read_matrices('livingRoom2n.gt.sim').to(device)

original_intrinsics = torch.tensor([
    [481.20, 0, 319.50],
    [0, -480.00, 239.50],
    [0, 0, 1],
]).to(device)

height = 100
width = 100

intrinsics = torch.tensor([
    [481.20/640*100, 0, 49.50],
    [0, -480.00/480*100, 49.50],
    [0, 0, 1],
]).to(device)

n_train = 150
# images = torch.zeros((n_train, height, width, 3))

def read_img_to_tensor(index, height, width):
    raw_img = cv2.imread(f'icl_nuim/scene_00_{index:04}.png') / 255.
    res_img = cv2.resize(raw_img, (height, width))
    return torch.from_numpy(res_img).float()


#train_idcs = torch.randint(0, all_poses.shape[0], (n_train,))
train_idcs = torch.arange(0, n_train).to(device)
images = torch.stack([
    read_img_to_tensor(tri, height, width) \
    for tri in train_idcs
]).to(device)

plt.figure()
plt.imshow(images[0].cpu().numpy())
plt.axis('off')
plt.title(f'Training sample {train_idcs[0]}')
plt.savefig(f'nuke/icl_train_vis_{0}.png')

poses = all_poses[train_idcs]

test_image = read_img_to_tensor(n_train+1, height, width).to(device)
test_pose = all_poses[n_train+1]


def analyze_depths(file_path):
    with open(file_path, 'r') as file:
        # Read the single line of float values
        values_line = file.readline().strip()

        # Convert the space-separated values into a NumPy array
        values = np.array(list(map(float, values_line.split())))

        # Calculate and print the statistics
        max_value = np.max(values)
        min_value = np.min(values)
        avg_value = np.mean(values)
        std_dev = np.std(values)

        print(f'Maximum: {max_value:.4f}')
        print(f'Minimum: {min_value:.4f}')
        print(f'Average: {avg_value:.4f}')
        print(f'Standard Deviation: {std_dev:.4f}')

analyze_depths('icl_nuim/scene_00_0150.depth')

num_x_frequencies = 15
num_d_frequencies = 6
learning_rate  = 7.5e-4
iterations = 3000
samples = 100
display = 50
near = 2
far = 6
model = nerf_model(num_x_frequencies=num_x_frequencies,num_d_frequencies=num_d_frequencies).to(device)
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
psnrs = []
iternums = []
criterion = torch.nn.MSELoss()

def train(model, iterations):
    tag = 'v2'
    
    t = time.time()
    t0 = time.time()
    for i in range(iterations+1):

       
        output_image_idx = np.random.choice(images.shape[0])
        output_image = images[output_image_idx,...]
        pose = poses[output_image_idx,...]
        # Run one iteration of NeRF and get the rendered RGB image.
        optimizer.zero_grad()
        rec_image = one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies)

        loss = criterion(rec_image, output_image)
        loss.backward()
        optimizer.step()


       
        if i % display == 0:
            with torch.no_grad():
           
                test_rec_image = one_forward_pass(height, width, intrinsics, test_pose, near, far, samples, model, num_x_frequencies, num_d_frequencies)

                
                loss = criterion(test_rec_image, test_image)
            psnr = 10 * torch.log10((torch.max(test_image)**2)/loss.item())


            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f " % psnr.item(), \
                    "Time: %.2f secs per iter, " % ((time.time() - t) / display), "%.2f mins in total" % ((time.time() - t0)/60))

            t = time.time()
            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(16, 4))
            plt.subplot(141)
            plt.imshow(test_rec_image.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(142)
            plt.imshow(test_image.detach().cpu().numpy())
            plt.title("Target image")
            plt.subplot(143)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.savefig(os.path.join(vis_path, f'icl_nuim_training_interm_{tag}_{i}.png'))
            plt.close()
            # plt.show()

    plt.imsave('test_lego.png',test_rec_image.detach().cpu().numpy())
    torch.save(model.state_dict(), f'icl_nuim_model_nerf_{tag}.pt')
    print('Done!')
    
if __name__ == '__main__':
    train(model, 2000)