import h5py
import json
import numpy as np
import open3d as o3d
import os
import trimesh

class IO:
    @classmethod
    def get(cls, file_path, with_cur=False):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.obj', '.stl', '.ply']:
            return cls._read_obj(file_path)
        elif file_extension in ['.json']:
            return cls._read_json(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = o3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
    
    @classmethod
    def _read_obj(cls, file_path, with_cur=False):
        mesh = trimesh.load(file_path, process=False)
        pts = np.asarray(mesh.vertices).astype(np.float32)
        if with_cur:
            curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=0.1)
            curvature = np.array(curvature).astype(np.float32)
            return pts, curvature
        else:
            return pts
    
    @classmethod
    def _read_json(cls, file_path):
        with open(file_path, 'r') as f:
            load_dict = json.load(f)
        return load_dict
 