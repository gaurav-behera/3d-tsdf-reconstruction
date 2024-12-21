__shared__ float shared_inv_frame_pose[12];
__shared__ float shared_cam_intrinsics[9];


__global__ void add_frame_gpu(float* tsdf, float* tsdf_weight, unsigned char* tsdf_color, float* frame_color_img, float* frame_depth_img, float* inv_frame_pose, float* cam_intrinsics, float* volume_dims, float* volume_origin, float* params)
{
    // read the parameters
    float weight = params[0];
    float voxel_size = params[1];
    float trunc_margin = params[2];
    int height = int(params[3]);
    int width = int(params[4]);

    // get indices
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_k = blockIdx.z * blockDim.z + threadIdx.z;
    int voxel_idx = idx_i * blockDim.y * blockDim.z + idx_j * blockDim.z + idx_k;

    // save to shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int i = 0; i < 12; i++) {
            shared_inv_frame_pose[i] = inv_frame_pose[i];
        }
        for (int i = 0; i < 9; i++) {
            shared_cam_intrinsics[i] = cam_intrinsics[i];
        }
    }
    // Ensure all threads wait until shared memory is initialized
    __syncthreads();

    // check if the indices are within the volume
    if (voxel_idx >= volume_dims[0] * volume_dims[1] * volume_dims[2])
        return;

    // x,y,z voxel coordinates
    int x = voxel_idx / (volume_dims[1] * volume_dims[2]);
    int y = (voxel_idx - x * volume_dims[1] * volume_dims[2]) / volume_dims[2];
    int z = voxel_idx - x * volume_dims[1] * volume_dims[2] - y * volume_dims[2];

    // voxel in world coordinates
    float voxel_world_x = volume_origin[0] + x * voxel_size;
    float voxel_world_y = volume_origin[1] + y * voxel_size;
    float voxel_world_z = volume_origin[2] + z * voxel_size;

    // voxel in camera coordinates
    float voxel_cam_x = shared_inv_frame_pose[0] * voxel_world_x + shared_inv_frame_pose[1] * voxel_world_y + shared_inv_frame_pose[2] * voxel_world_z + shared_inv_frame_pose[3];
    float voxel_cam_y = shared_inv_frame_pose[4] * voxel_world_x + shared_inv_frame_pose[5] * voxel_world_y + shared_inv_frame_pose[6] * voxel_world_z + shared_inv_frame_pose[7];
    float voxel_cam_z = shared_inv_frame_pose[8] * voxel_world_x + shared_inv_frame_pose[9] * voxel_world_y + shared_inv_frame_pose[10] * voxel_world_z + shared_inv_frame_pose[11];

    // pixel coordinates
    int pixel_x = (int) roundf((shared_cam_intrinsics[0] * voxel_cam_x + shared_cam_intrinsics[1] * voxel_cam_y + shared_cam_intrinsics[2]* voxel_cam_z) / (voxel_cam_z * shared_cam_intrinsics[8]));
    int pixel_y = (int) roundf((shared_cam_intrinsics[3] * voxel_cam_x + shared_cam_intrinsics[4] * voxel_cam_y + shared_cam_intrinsics[5] * voxel_cam_z) / (voxel_cam_z * shared_cam_intrinsics[8]));

    // check if the pixel is within the frame
    if (pixel_x < 0 || pixel_x >= width || pixel_y < 0 || pixel_y >= height || voxel_cam_z < 0)
        return;
    
    // get the depth value
    float depth_val = frame_depth_img[pixel_y * width + pixel_x];
    float sdf_val = depth_val - voxel_cam_z;
    if (depth_val == 0 || sdf_val < -trunc_margin)
        return;

    float tsdf_val = fmin(1.0f, sdf_val / trunc_margin);

    // retrieve the old values
    float old_tsdf = tsdf[voxel_idx];   
    float old_weight = tsdf_weight[voxel_idx];
    float old_color_r = float(tsdf_color[voxel_idx*3+0]);
    float old_color_g = float(tsdf_color[voxel_idx*3+1]);
    float old_color_b = float(tsdf_color[voxel_idx*3+2]);

    // update the tsdf value
    float new_weight = old_weight + weight;
    tsdf[voxel_idx] = (old_tsdf * old_weight + tsdf_val * weight) / new_weight;
    tsdf_weight[voxel_idx] = new_weight;
    tsdf_color[voxel_idx*3+0] = (old_color_r * old_weight + frame_color_img[(pixel_y * width + pixel_x)*3+0] * weight) / new_weight;
    tsdf_color[voxel_idx*3+1] = (old_color_g * old_weight + frame_color_img[(pixel_y * width + pixel_x)*3+1] * weight) / new_weight;
    tsdf_color[voxel_idx*3+2] = (old_color_b * old_weight + frame_color_img[(pixel_y * width + pixel_x)*3+2] * weight) / new_weight;    
}
