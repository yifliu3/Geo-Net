import trimesh
import os
from tqdm import tqdm
import numpy as np

raw_file_path = '/data/yfliu/full/scans'
filename_list = os.listdir(raw_file_path)

for filename in tqdm(filename_list):
    path_cur = os.path.join(raw_file_path, filename)
    lower_obj_path = os.path.join(raw_file_path, filename, 'lowerjaw.stl')
    if not os.path.exists(lower_obj_path):
        lower_obj_path = lower_obj_path.replace('.stl', '.ply')
    
    upper_obj_path = os.path.join(raw_file_path, filename, 'upperjaw.stl')
    if not os.path.exists(upper_obj_path):
        upper_obj_path = upper_obj_path.replace('.stl', '.ply')

    for i, path in enumerate([lower_obj_path, upper_obj_path]):
        mesh = trimesh.load(path, process=False)
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=0.1)
        curvature = np.array(curvature).astype(np.float32)
        position = 'lowercur' if i == 0 else 'uppercur'
        np.save(f'{path_cur}/{position}.npy', curvature)

