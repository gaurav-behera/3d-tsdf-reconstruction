import numpy as np
import matplotlib.pyplot as plt
import mcubes
import trimesh

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    print("PyCUDA successfully imported. Running in GPU mode.")
except Exception as err:
    print("Failed to import PyCUDA. Running in CPU mode.")


class TSDF:
    def __init__(self, volume_bounds, voxel_size, trunc_margin, run_on_gpu=False, verbose=True):
        """
        volume_bounds -- the min and max bounds of the volume
        voxel_size -- the size of each voxel
        trunc_margin -- the truncation margin
        """
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin
        self.volume_bounds = volume_bounds
        self.run_on_gpu = run_on_gpu
        # calculate the dimensions of the volume
        self.volume_dims = np.ceil(
            (volume_bounds[:, 1] - volume_bounds[:, 0]) / voxel_size
        ).astype(int)

        # update the volume bounds
        self.volume_bounds[:, 1] = (
            self.volume_bounds[:, 0] + self.volume_dims * voxel_size
        )
        self.volume_center = (self.volume_bounds[:, 0] + self.volume_bounds[:, 1]) / 2
        self.volume_origin = self.volume_center - self.volume_dims * voxel_size / 2
        self.verbose = verbose
        
        if self.verbose:
            print("Volume dims: ", self.volume_dims)

        if not self.run_on_gpu:
            # store voxel coords
            x, y, z = np.meshgrid(
                range(self.volume_dims[0]),
                range(self.volume_dims[1]),
                range(self.volume_dims[2]),
                indexing="ij",
            )
            self.voxel_coords = np.stack([x, y, z], axis=3).reshape(-1, 3)
            # world voxel coordinates
            world_voxel_coords = (
                self.voxel_coords * self.voxel_size + self.volume_origin
            )
            self.world_voxel_coords = np.hstack(
                [world_voxel_coords, np.ones((world_voxel_coords.shape[0], 1))]
            )

        # initialize the tsdf
        self.tsdf = np.ones(self.volume_dims).astype(
            np.float32
        )  # truncated signed distance
        self.weight = np.zeros(self.volume_dims).astype(
            np.float32
        )  # uncertainty of each voxel
        self.color = np.zeros(self.volume_dims.tolist() + [3]).astype(
            np.uint8
        )  # color of each voxel

        if self.run_on_gpu:
            self.init_gpu()

    def init_gpu(self):
        """
        Initialize the GPU for running the TSDF algorithm.
        """
        # allocate memory on the GPU
        self.tsdf_gpu = cuda.mem_alloc(self.tsdf.nbytes)
        self.weight_gpu = cuda.mem_alloc(self.weight.nbytes)
        self.color_gpu = cuda.mem_alloc(self.color.nbytes)

        # copy data to the GPU
        cuda.memcpy_htod(self.tsdf_gpu, self.tsdf)
        cuda.memcpy_htod(self.weight_gpu, self.weight)
        cuda.memcpy_htod(self.color_gpu, self.color)

        # compile the CUDA kernel
        with open("add_frame_gpu.cu", "r") as f:
            mod = SourceModule(
                f.read(), options=["--compiler-bindir", "/usr/bin/g++-9"]
            )
        self.add_frame_gpu = mod.get_function("add_frame_gpu")

        # gpu device details
        gpu_device = pycuda.autoinit.device
        if self.verbose:
            print("Using GPU: ", gpu_device.name())
            print(
                f"Maximum number of threads per block: {gpu_device.MAX_THREADS_PER_BLOCK}"
            )
            print(
                f"Maximum block dimension: {gpu_device.MAX_BLOCK_DIM_X}, {gpu_device.MAX_BLOCK_DIM_Y}, {gpu_device.MAX_BLOCK_DIM_Z}"
            )
            print(
                f"Maximum grid dimension: {gpu_device.MAX_GRID_DIM_X}, {gpu_device.MAX_GRID_DIM_Y}, {gpu_device.MAX_GRID_DIM_Z}"
            )

        # set the block and grid size
        self.block_size = (int(gpu_device.MAX_THREADS_PER_BLOCK), int(1), int(1))
        total_threads = np.prod(self.volume_dims)
        grid_size_x = min(
            gpu_device.MAX_GRID_DIM_X, int(np.ceil(total_threads / self.block_size[0]))
        )
        rem_threads = max(total_threads - grid_size_x * self.block_size[0], 1)
        grid_size_y = min(
            gpu_device.MAX_GRID_DIM_Y, int(np.ceil(rem_threads / self.block_size[1]))
        )
        rem_threads = max(rem_threads - grid_size_y * self.block_size[1], 1)
        grid_size_z = min(
            gpu_device.MAX_GRID_DIM_Z, int(np.ceil(rem_threads / self.block_size[2]))
        )
        self.grid_size = (grid_size_x, grid_size_y, grid_size_z)
        if self.verbose:
            print(f"Block size: {self.block_size}, Grid size: {self.grid_size}")

    def add_frame(
        self, frame_color_img, frame_depth_img, frame_pose, cam_intrinsics, weight=1
    ):
        """
        Add a frame to the TSDF volume.
        frame_color_img -- the color image of the frame
        frame_depth_img -- the depth image of the frame
        frame_pose -- the pose of the frame
        cam_intrinsics -- the camera intrinsics
        """
        height, width = frame_depth_img.shape
        inv_frame_pose = np.linalg.inv(frame_pose).astype(np.float32)

        if self.run_on_gpu:
            # call the CUDA kernel function
            grid_tuple = tuple(self.grid_size)
            block_tuple = tuple(self.block_size)
            self.add_frame_gpu(
                self.tsdf_gpu,
                self.weight_gpu,
                self.color_gpu,
                cuda.InOut(frame_color_img.flatten().astype(np.float32)),
                cuda.InOut(frame_depth_img.flatten().astype(np.float32)),
                cuda.InOut(inv_frame_pose.flatten()),
                cuda.InOut(cam_intrinsics.flatten().astype(np.float32)),
                cuda.InOut(self.volume_dims.astype(np.float32)),
                cuda.InOut(self.volume_origin.astype(np.float32)),
                cuda.InOut(
                    np.array(
                        [weight, self.voxel_size, self.trunc_margin, height, width],
                        dtype=np.float32,
                    )
                ),
                block=block_tuple,
                grid=grid_tuple,
            )
        else:
            # camera voxel coordinates
            cam_voxel_coords = np.dot(inv_frame_pose, self.world_voxel_coords.T).T[
                :, :3
            ]
            # pixel coordinates
            pixel_coords = np.dot(cam_intrinsics, cam_voxel_coords.T).T
            pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:]  # normalize
            pixel_coords = pixel_coords.astype(int)

            # valid pixel coordinates based on image width and height
            valid_pixel_coords = np.all(
                np.logical_and(pixel_coords >= 0, pixel_coords < [width, height]),
                axis=1,
            )

            # depth map
            depth_vals = np.zeros(len(self.world_voxel_coords))
            depth_vals[valid_pixel_coords] = frame_depth_img[
                pixel_coords[valid_pixel_coords][:, 1],
                pixel_coords[valid_pixel_coords][:, 0],
            ]

            # sdf values
            sdf_vals = depth_vals - cam_voxel_coords[:, 2]
            # valid voxel coordinates are only the voxels that have positive depth values and sdf value greater than -trunc_margin
            valid_voxel_coords = np.logical_and(
                depth_vals > 0, sdf_vals >= -self.trunc_margin
            )  # mask
            valid_voxels_x = self.voxel_coords[valid_voxel_coords][:, 0]
            valid_voxels_y = self.voxel_coords[valid_voxel_coords][:, 1]
            valid_voxels_z = self.voxel_coords[valid_voxel_coords][:, 2]
            valid_pixel_coords_x = pixel_coords[:, 0][valid_voxel_coords]
            valid_pixel_coords_y = pixel_coords[:, 1][valid_voxel_coords]
            # truncate sdf values
            sdf_vals = np.clip(sdf_vals / self.trunc_margin, -1, 1)

            # retrive details of only the valid voxel coordinates (saves time)
            current_weights = self.weight[
                valid_voxels_x, valid_voxels_y, valid_voxels_z
            ]
            current_tsdf = self.tsdf[valid_voxels_x, valid_voxels_y, valid_voxels_z]
            current_color = self.color[valid_voxels_x, valid_voxels_y, valid_voxels_z]
            observed_sdf_vals = sdf_vals[valid_voxel_coords]
            observed_color_vals = frame_color_img[
                valid_pixel_coords_y, valid_pixel_coords_x
            ]

            # compute the new tsdf and weights
            new_weight = current_weights + weight
            new_tsdf = (
                current_tsdf * current_weights + observed_sdf_vals * weight
            ) / new_weight
            new_color = (
                current_color
                * np.stack([current_weights, current_weights, current_weights], axis=-1)
                + observed_color_vals * np.stack([weight, weight, weight], axis=-1)
            ) / np.stack([new_weight, new_weight, new_weight], axis=-1)

            # update the tsdf and weights
            self.tsdf[valid_voxels_x, valid_voxels_y, valid_voxels_z] = new_tsdf
            self.weight[valid_voxels_x, valid_voxels_y, valid_voxels_z] = new_weight
            self.color[valid_voxels_x, valid_voxels_y, valid_voxels_z] = new_color

    def build_tsdf(
        self,
        frame_count,
        frame_color_imgs,
        frame_depth_imgs,
        frame_poses,
        cam_intrinsics,
    ):
        for i in range(frame_count):
            if self.verbose:
                print("Fusing frame", i)
            self.add_frame(
                frame_color_imgs[i], frame_depth_imgs[i], frame_poses[i], cam_intrinsics
            )
            
    def get_tsdf(self):
        """
        Return the TSDF volume.
        """
        if self.run_on_gpu:
            # copy data from the GPU
            cuda.memcpy_dtoh(self.tsdf, self.tsdf_gpu)
            cuda.memcpy_dtoh(self.weight, self.weight_gpu)
            cuda.memcpy_dtoh(self.color, self.color_gpu)
        return self.tsdf, self.weight, self.color
            
    def marching_cubes(self):
        """
        Return the vertices and triangles using marching cubes.
        """
        if self.run_on_gpu:
            # copy data from the GPU
            cuda.memcpy_dtoh(self.tsdf, self.tsdf_gpu)
            cuda.memcpy_dtoh(self.weight, self.weight_gpu)
            cuda.memcpy_dtoh(self.color, self.color_gpu)
        vertices, triangles = mcubes.marching_cubes(self.tsdf, 0)
        return vertices, triangles
    
    def save_mesh(self, file_name="tsdf_mesh.ply"):
        """
        Generate a mesh from the TSDF volume.
        """
        vertices, triangles = self.marching_cubes()
        cvx, cvy, cvz = np.round(vertices).astype(int).T
        vertices_world = vertices * self.voxel_size + self.volume_origin
        colors = self.color[cvx, cvy, cvz]
        mesh = trimesh.Trimesh(vertices_world, triangles, vertex_colors=colors)
        mesh.export(file_name)
        print(f"Mesh saved to {file_name}")

    def save_point_cloud(self, file_name="tsdf_point_cloud.ply"):
        """
        Generate a point cloud from the TSDF volume.
        """
        vertices, triangles = self.marching_cubes()
        cvx, cvy, cvz = np.round(vertices).astype(int).T
        vertices_world = vertices * self.voxel_size + self.volume_origin
        colors = self.color[cvx, cvy, cvz]
        point_cloud = np.hstack([vertices_world, colors])
        header = (
            f"ply\n"
            f"format ascii 1.0\n"
            f"element vertex {point_cloud.shape[0]}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"end_header\n"
        )

        np.savetxt(
            file_name,
            point_cloud,
            fmt="%.6f %.6f %.6f %d %d %d",
            header=header,
            comments="",
        )
        print(f"Point cloud saved to {file_name}")


def view_frustrum_limits(frame_pose, frame_depth_img, inv_cam_intrinsics):
    """
    Calculate the 3D view frustum of a frame in world coordinates and returns the limits of the frustum.
    """
    height, width = frame_depth_img.shape
    max_depth = frame_depth_img.max()

    # corner points in image coordinates
    pixel_corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )

    camera_corners = []  # points in camera coordinates
    for px, py in pixel_corners:
        normalized_point = np.dot(inv_cam_intrinsics, np.array([px, py, 1]))
        camera_corners.append(normalized_point * max_depth)
    camera_corners = np.array(camera_corners)

    camera_corners = np.vstack([camera_corners, np.zeros((1, 3))])  # camera origin

    # transform to world coordinates
    homogeneous_corners = np.hstack([camera_corners, np.ones((5, 1))])
    world_corners = np.dot(frame_pose, homogeneous_corners.T).T[:, :3]

    # calculate the limits of the frustum
    min_bounds = world_corners.min(axis=0)
    max_bounds = world_corners.max(axis=0)
    return min_bounds, max_bounds


def get_volume_bounds(frame_count, frame_depth_imgs, frame_poses, cam_intrinsics):
    """
    Calculate the bounds of the TSDF volume based on the camera frustum of all the frames.
    """
    volume_bounds = np.zeros((3, 2)) # min and max bounds
    volume_bounds[:, 0] = np.inf
    volume_bounds[:, 1] = -np.inf
    inv_cam_intrinsics = np.linalg.inv(cam_intrinsics)
    for i in range(frame_count):
        min_bounds, max_bounds = view_frustrum_limits(frame_poses[i], frame_depth_imgs[i], inv_cam_intrinsics)
        volume_bounds[:, 0] = np.minimum(volume_bounds[:, 0], min_bounds)
        volume_bounds[:, 1] = np.maximum(volume_bounds[:, 1], max_bounds)

    print("Volume bounds:")
    print(volume_bounds)
    return volume_bounds
