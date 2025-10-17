import os
import cv2
import pyzed.sl as sl
import shutil
import argparse
import time
import numpy as np


def create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def construct_camera_intrinsics(camera_params, target_width, target_height, camera_upsidedown=False):
    """
    Construct a 3x3 camera intrinsics matrix from ZED camera parameters.
    
    Args:
        camera_params: ZED CameraParameters object
        target_width: Target image width after resizing
        target_height: Target image height after resizing
        camera_upsidedown: Whether the camera is mounted upside down
    
    Returns:
        3x3 numpy array representing the camera intrinsics matrix K
    """
    # Get original parameters
    fx_orig = camera_params.fx
    fy_orig = camera_params.fy
    cx_orig = camera_params.cx
    cy_orig = camera_params.cy
    
    # Get original resolution
    orig_width = camera_params.image_size.width
    orig_height = camera_params.image_size.height
    
    # Scale intrinsics to target resolution
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y
    
    # If camera is upside down (180 degree rotation), adjust principal point
    if camera_upsidedown:
        cx = target_width - cx
        cy = target_height - cy
    
    # Construct 3x3 intrinsics matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save captured frames')
    parser.add_argument('--serial_number', type=str, default='31304053', help='ZED camera serial number')
    parser.add_argument('--camera_upsidedown', action='store_true', help='Whether camera is mounted upside down')
    parser.add_argument('--width', type=int, default=960, help='Image width')
    parser.add_argument('--height', type=int, default=540, help='Image height')
    parser.add_argument('--exposure', type=int, default=25, help='Camera exposure value')
    parser.add_argument('--gain', type=int, default=40, help='Camera gain value')
    parser.add_argument('--fps', type=float, default=10.0, help='Frames per second for capture (default: 10)')
    
    args = parser.parse_args()
    
    # Calculate target frame time from FPS
    target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.1
    
    serial_number = args.serial_number
    camera_upsidedown = args.camera_upsidedown
    
    zed = sl.Camera()
    input_type = sl.InputType()
    # input_type.set_from_svo_file("verify_camera_demo/one_frame.svo2")
    init_params = sl.InitParameters(input_t=input_type)
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        exit(1)

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, args.exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, args.gain)
    
    # Create a directory to save frames
    save_directory = args.save_dir

    rgb_directory = os.path.join(save_directory, "rgb")
    depth_directory = os.path.join(save_directory, "depth")

    create_directory(rgb_directory)
    create_directory(depth_directory)
    
    # Get camera parameters and construct intrinsics matrix
    camera_info = zed.get_camera_information()
    left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
    
    K = construct_camera_intrinsics(
        left_cam_params,
        args.width,
        args.height,
        camera_upsidedown
    )
    
    # Save camera intrinsics to file
    intrinsics_path = os.path.join(save_directory, "cam_K.txt")
    np.savetxt(intrinsics_path, K)
    print(f"Camera intrinsics saved to {intrinsics_path}")
    print(f"K matrix:\n{K}")

    runtime_parameters = sl.RuntimeParameters()
    mat = sl.Mat()
    depth_mat = sl.Mat()

    frame_count = 0

    try:
        while True:
            frame_start_time = time.time()
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

                # Retrieve RGB frame
                zed.retrieve_image(mat, sl.VIEW.LEFT)
                rgb_image = mat.get_data()
                
                if camera_upsidedown:
                    rgb_image = cv2.flip(rgb_image, -1)
                rgb_image = cv2.resize(rgb_image, (args.width, args.height))

                cv2.imshow("RGB", rgb_image)
                
                # Check for spacebar to stop recording
                k = cv2.waitKey(1) & 0xFF
                if k == 32:
                    break

                rgb_filename = os.path.join(
                    rgb_directory, f"frame_{frame_count:04d}.png"
                )
                cv2.imwrite(rgb_filename, rgb_image)


                # Retrieve depth frame
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_image = depth_mat.get_data()

                depth_image = (depth_image).astype("uint16")   
                if camera_upsidedown:
                    depth_image = cv2.flip(depth_image, -1)
                depth_image = cv2.resize(depth_image, (args.width, args.height))  

                depth_filename = os.path.join(
                    depth_directory, f"frame_{frame_count:04d}.png"
                )
                cv2.imwrite(depth_filename, depth_image)

                print(f'Frame: {frame_count}')
                frame_count += 1
        
            else:   
                break

            # Calculate elapsed time and sleep to match target FPS
            frame_elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - frame_elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping frame capture")

    finally:
        # Release the camera
        zed.close()
        cv2.destroyAllWindows()
        print(f"Captured {frame_count} frames to {save_directory}")
