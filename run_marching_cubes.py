import tqdm
from tqdm.contrib.concurrent import process_map

# marching cube using skimage
from skimage.measure import marching_cubes
import skimage
import glob
import numpy as np
import trimesh

TSDF_VALUE = 1/32
SDF_CLIP_VALUE = 0.05

def run_marching_cubes(args):
    """
    Perform marching cubes on the occupancy array.
    Returns vertices, faces, normals, and values.
    """
    occ_arr, save_path = args
    verts, faces, _, _ = marching_cubes(occ_arr, level=0.5)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(save_path)
    print("Saved mesh to", save_path)

if __name__ == '__main__':

    dataset_folder = '/data/octfusion_sdf_data'
    resolution = 128

    multiplier = 128 // resolution

    save_folder = '/data/octfusion_sdf_data/low' if multiplier == 2 else '/data/octfusion_sdf_data/high'


    # Get all the .npy files in the dataset folder
    sdf_files = glob.glob(f"{dataset_folder}/*.npy")
    print(f"Found {len(sdf_files)} files in {dataset_folder}")

    # load each file
    occ_arrs = []
    for sdf_file in tqdm.tqdm(sdf_files):
        print(f"Processing {sdf_file}")
        # Load the SDF file
        sdf = np.load(sdf_file)

        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (multiplier, multiplier, multiplier), np.max)


        save_path = f"{save_folder}/{sdf_file.split('/')[-1]}".replace('.npy', '.obj')

        occ_arrs.append((occupancy_low, save_path))

    process_map(run_marching_cubes, occ_arrs, max_workers=32)


