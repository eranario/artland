from math import degrees
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from pupil_apriltags import Detector
from PIL import Image as im

# Camera Intrinsics and Tag Size (848x480 res)
fx = 427.18
fy = 427.18
cx = 425.013
cy = 239.419
params = [fx, fy, cx, cy]
size = 0.137 # in meters

# Initialize Detector
detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Detect tags
        color_image_data = color_image[:,:,0]
        data = im.fromarray(color_image_data)
        data.save('data.png')
        image = cv2.imread('data.png')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=size)
        # print("[INFO] {} total AprilTags detected".format(len(tags)))

        # Draw the bounding box of the AprilTag detection
        for t in tags:

            (ptA, ptB, ptC, ptD) = t.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)

            (cX, cY) = (int(t.center[0]), int(t.center[1]))
            cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

            tagID = "tag ID: " + str(t.tag_id) #.decode("utf-8")
            cv2.putText(color_image, tagID, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Obtain tag attributes
        index = len(tags)-1
        if index >= 0:
            while index >= 0:
                t_matrix = tags[index].pose_t
                dist = (t_matrix[0]**2 + t_matrix[1]**2 + t_matrix[2]**2)**0.5
                # print(dist)

                r_matrix = tags[index].pose_R
                angle = r_matrix('xyz',degrees=True)
                print(angle)

                index = index - 1

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
