import open3d as o3d
import numpy as np
import torch
import copy

from extensions.pointops.functions import pointops

data_path = '/home/yfliu/data/3dTeethSeg/scans/0EAKT1CU/0EAKT1CU_upper.obj'

mesh = o3d.io.read_triangle_mesh(data_path)
pcs = np.asarray(mesh.vertices).astype(np.float32)

num_avail_pts = pcs.shape[0]
idxs = np.arange(num_avail_pts)
np.random.shuffle(idxs)
sampled_pcs = pcs[idxs[:16000], :]

center = sampled_pcs[0]
center_tensor = copy.deepcopy(torch.from_numpy(center).unsqueeze(0).unsqueeze(0).cuda())
sampled_pcs_tensor = copy.deepcopy(torch.from_numpy(sampled_pcs).unsqueeze(0).cuda())

idx = pointops.knn(center_tensor, sampled_pcs_tensor, 256)[0] # B G M
neighborhood = pointops.index_points(sampled_pcs_tensor, idx) # B G M 3
neighborhood = neighborhood.squeeze().detach().cpu().numpy()

pt = o3d.geometry.PointCloud()
pt.points = o3d.utility.Vector3dVector(neighborhood)

o3d.visualization.draw_geometries([pt], window_name='test', width=500, height=500)