import time
import cv2
import numpy as np
from tsdf import TSDF, get_volume_bounds

# define data paths
data_folder = "data/kitchen-7scenes/"
cam_intrinsics_file = data_folder + "camera-intrinsics.txt"
frame_color_img_file = data_folder + "frame-{:06d}.color.jpg"
frame_depth_img_file = data_folder + "frame-{:06d}.depth.png"
frame_pose_file = data_folder + "frame-{:06d}.pose.txt"
frames_count = 1000


def process_data():
    frame_color_imgs = []
    frame_depth_imgs = []
    frame_poses = []

    # read the data
    for i in range(frames_count):
        color_img = cv2.imread(frame_color_img_file.format(i))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        frame_color_imgs.append(color_img)
        depth_img = (
            cv2.imread(frame_depth_img_file.format(i), cv2.IMREAD_UNCHANGED) / 1000
        )
        depth_img[depth_img == 65.535] = 0
        frame_depth_imgs.append(depth_img)
        frame_poses.append(np.loadtxt(frame_pose_file.format(i)))
    return frame_color_imgs, frame_depth_imgs, frame_poses


frame_color_imgs, frame_depth_imgs, frame_poses = process_data()
cam_intrinsics = np.loadtxt(cam_intrinsics_file)

# tsdf
tsdf_voxel_size = 0.02  # 2cm
tsdf_trunc_margin = 0.1  # 10cm

volume_bounds = get_volume_bounds(
    frames_count, frame_depth_imgs, frame_poses, cam_intrinsics
)

tsdf_obj = TSDF(
    volume_bounds,
    voxel_size=tsdf_voxel_size,
    trunc_margin=tsdf_trunc_margin,
    run_on_gpu=True,
)
start_time = time.time()
tsdf_obj.build_tsdf(
    frames_count, frame_color_imgs, frame_depth_imgs, frame_poses, cam_intrinsics
)
end_time = time.time()
print("Time: ", end_time - start_time, "seconds")
print("FPS: ", frames_count / (end_time - start_time))

tsdf_obj.save_mesh("outputs/kitchen_mesh.ply")
tsdf_obj.save_point_cloud("outputs/kitchen_pc.ply")
