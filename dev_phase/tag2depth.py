# use bounding boxes from apriltag detection to extra depth pixels for orientation

import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector
from PIL import Image as im
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pyransac3d as pyrsc
import math
import csv

# Camera Intrinsics
fx = 644.799
fy = 644.799
cx = 641.529
cy = 359.124
params = [fx, fy, cx, cy]
size = 0.155 # in meters

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

# RGB information
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# align RGB and depth images
align_to = rs.stream.color
align = rs.align(align_to)

# DEPTH information
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# storing calculated tag attributes into this class
class Tag:
    def __init__(self, t_dist, t_angle, d_dist, d_angle, tagID):
        self.t_dist = t_dist
        self.t_angle = t_angle
        self.d_dist = d_dist
        self.d_angle = d_angle
        self.ID = tagID

    def get_coord(self):
        return

    def get_gpsc(self):
        return

# volatile tag attributes
t1 = Tag(0,0,0,0,0)
t2 = Tag(0,0,0,0,0)
t3 = Tag(0,0,0,0,0)
t4 = Tag(0,0,0,0,0)
t5 = Tag(0,0,0,0,0)
t6 = Tag(0,0,0,0,0)
t7 = Tag(0,0,0,0,0)
t8 = Tag(0,0,0,0,0)

try:
    f = open('D:/artland/data/spreadsheets/tag_depth_comp.csv', 'a', encoding='UTF8', newline='')
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        mask = np.zeros(depth_image.shape, dtype="uint8")
        newDepth = np.zeros(depth_image.shape, dtype="uint8")

        # Detect tags
        color_image_data = color_image[..., ::-1]
        data = im.fromarray(color_image_data)
        data.save('data.png')
        image = cv2.imread('data.png')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=size)
        # print("[INFO] {} total AprilTags detected".format(len(tags)))

        # volatile data storage
        memory = [t1,t2,t3,t4,t5,t6,t7,t8]

        # Draw the bounding box of the AprilTag detection
        index = 0
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
            
            # draw box around RGB images
            tagID = "tag ID: " + str(t.tag_id) #.decode("utf-8")
            cv2.putText(color_image, tagID, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # extract deopth pixels from bounding box
            corners = np.array([list(ptA),list(ptB),list(ptC),list(ptD)])
            cv2.fillPoly(mask,pts=[corners],color=(255,255,255))
            tagDepth = cv2.bitwise_and(depth_image,depth_image,mask = mask)
            newDepth = cv2.bitwise_and(depth_image,depth_image,mask = mask)

            # calculate distances and angle from tag info
            t_matrix = t.pose_t
            t_dist = (t_matrix[0]**2 + t_matrix[1]**2 + t_matrix[2]**2)**0.5
            
            r_matrix = t.pose_R
            r = R.from_matrix(r_matrix)
            angle = np.asarray(r.as_euler('zxy',degrees=True)) # roll pitch yaw
            t_angle = np.abs(angle[2])

            # calculate distances and angle from depth info
            d_dist = tagDepth[np.nonzero(tagDepth)].mean()*depth_scale

            depth_raw = o3d.geometry.Image(tagDepth)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(848,480,fx,fy,cx,cy)
            pc = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,intrinsic)

            points = np.asarray(pc.points)
            plane = pyrsc.Plane()
            best_eq, best_inliers = plane.fit(points,0.01)
            n = np.array(best_eq[0:3])
            origin = np.array([0,0,1])
            nn = np.dot(n,origin) / (np.linalg.norm(n)*origin)
            d_angle = np.abs((np.arccos(nn) * 180)) / math.pi

            # store memory
            memory[index] = Tag(t_dist,t_angle,d_dist,d_angle,tagID)

            # display on terminal
            # print(f'tag dist: {t_dist[0]},tag angle: {t_angle},depth dist: {d_dist},depth angle: {d_angle[2]} ,tag ID: {tagID}')

            # write into csv
            data = [tagID, t_dist[0], t_angle, d_dist, d_angle[2]]
            writer = csv.writer(f)
            writer.writerow(data)

            # reset/upgrade short-term variables
            tagDepth = 0
            index = index + 1
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(newDepth, alpha=0.03), cv2.COLORMAP_JET)

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

    f.close()