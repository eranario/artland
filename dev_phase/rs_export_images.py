import cv2
import imageio
import numpy as np
import pyrealsense2 as rs
import os
import shutil
from datetime import datetime

def make_clean_folder(path_folder):    
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == "y":
            shutil.rmtree(path_folder)
            os.mkdir(path_folder)
        else:
            exit()


def record_rgbd():
    date = datetime.now()
    make_clean_folder(f"data/{date.year}_{date.month}_{date.day}/")
    os.chdir("data")

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(
        rs.option.visual_preset, 3
    )  # Set high accuracy for depth sensor
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            date = datetime.now()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            grey_color = 153
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image)
            )  # Depth image is 1 channel, color is 3 channels
            bg_removed = np.where(
                (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                grey_color,
                color_image,
            )

            color_image = color_image[..., ::-1]

            ts = int(datetime.timestamp(date))

            if not os.path.exists(f"../data/{date.year}_{date.month}_{date.day}/depth_{date.year}_{date.month}_{date.day}_{ts}.png"):
                imageio.imwrite(f"../data/{date.year}_{date.month}_{date.day}/depth_{date.year}_{date.month}_{date.day}_{ts}.png", depth_image)
                imageio.imwrite(f"../data/{date.year}_{date.month}_{date.day}/rgb_{date.year}_{date.month}_{date.day}_{ts}.png", color_image)
            else:
                continue

    finally:
        pipeline.stop()

    return color_image, depth_image


if __name__ == "__main__":
    record_rgbd()