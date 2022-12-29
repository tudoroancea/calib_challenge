import numpy as np
import matplotlib.pyplot as plt

# read and plot data in labeled/*.txt
for i in range(5):
    data = np.loadtxt(f'labeled/{i}.txt')
    plt.figure()
    plt.plot(data[:,0], label='pitch')
    plt.plot(data[:,1], label='yaw')
    plt.title(f'Video {i}')
    plt.legend()
    plt.savefig(f'labeled/{i}.png')

plt.show()


# import torch
# import math
#
# def sift(image, scales):
#     """
#     Extracts SIFT features from an image using the scale-space representation of the image.
#
#     Parameters:
#     - image: a tensor of shape (C, H, W) representing the input image.
#     - scales: a list of scales to use for the Gaussian kernels in the DoG function.
#
#     Returns:
#     - features: a tensor of shape (num_features, 4) representing the SIFT features, where the columns
#       represent the locations, scales, and orientations of the features.
#     """
#     # Compute the scale-space representation of the image using the DoG function
#     dogs = []
#     for i in range(len(scales) - 1):
#         sigma1 = scales[i]
#         sigma2 = scales[i + 1]
#         dog = image - torch.nn.functional.gaussian_blur(image, sigma2) + torch.nn.functional.gaussian_blur(image, sigma1)
#         dogs.append(dog)
#
#     # Flatten the scale-space representation of the image
#     dogs = torch.cat(dogs, dim=0)
#     dogs = dogs.view(-1, dogs.shape[2], dogs.shape[3])
#
#     # Detect extrema in the scale-space representation of the image using a sliding window approach
#     features = []
#     for i in range(1, dogs.shape[0] - 1):
#         for j in range(1, dogs.shape[1] - 1):
#             for k in range(1, dogs.shape[2] - 1):
#                 center = dogs[i, j, k]
#                 neighbors = dogs[i-1:i+2, j-1:j+2, k-1:k+2].view(-1)
#                 is_extremum = True
#                 for neighbor in neighbors:
#                     if center > neighbor or center < neighbor:
#                         is_extremum = False
#                         break
#                 if is_extremum:
#                     # Assign a scale to the key point based on the scale of the Gaussian kernel that was used to compute the DoG
#                     scale = scales[i]
#
#                     # Assign an orientation to the key point based on the local gradient
#                     dx = image[:, j+1, k] - image[:, j-1, k]
#                     dy = image[:, j, k+1] - image[:, j, k-1]
#                     orientation = math.atan2(dy, dx)
#
#                     # Add the key point to the list of features
#                     features.append([j, k, scale, orientation])
#
#     # Return the list of features as a tensor
#     return torch.tensor(features)
#
#
# # read videos with torchvision.io.read_video
# import torchvision
# import torch.nn.functional as F
#
# # read and plot data in labeled/*.txt
# for i in range(1):
#     # read video
#     video, audio, info = torchvision.io.read_video(f'labeled/{i}.mp4')
#     # convert video to grayscale
#     video = F.rgb_to_grayscale(video)
#     # extract SIFT features
#     features = sift(video, [1, 2, 4, 8, 16])
#     # plot features
#     plt.figure()
#     plt.imshow(video[0,0,:,:], cmap='gray')
#     plt.scatter(features[:,1], features[:,0], s=1)
#     plt.title(f'Video {i}')
#
# plt.show()