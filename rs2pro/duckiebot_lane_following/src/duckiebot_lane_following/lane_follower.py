#!/usr/bin/env python

import numpy as np

import cv2

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from particle_filter import ParticleFilter

import functions

# Mask thresholds (HSV)
mask_lower_threshold = np.array([50, 16, 111])
mask_upper_threshold = np.array([95, 56, 153])

mask2_lower_threshold = np.array([28, 54, 88])
mask2_upper_threshold = np.array([84, 179, 169])


# Lane width
lane_width = 0.3


def plot_line_segments(image, line_segments):
    # Draw lines on image
    for line_segment in line_segments:
        u1, v1, u2, v2 = line_segment
        cv2.line(image, (int(u1), int(v1)), (int(u2), int(v2)), (255, 0, 255), 1, cv2.LINE_AA)
    print("plot_line_segments")
  

    return image


def plot_lanes(image, left_lane, right_lane):
    if left_lane is not None:
        [u1, v1, u2, v2] = left_lane
        cv2.line(image, (int(u1), int(v1)), (int(u2), int(v2)), (0, 0, 255), 2, cv2.LINE_AA)

    if right_lane is not None:
        [u1, v1, u2, v2] = right_lane
        cv2.line(image, (int(u1), int(v1)), (int(u2), int(v2)), (255, 0, 0), 2, cv2.LINE_AA)
    


    return image

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(1)

def estimate_pose(left_lane_image, right_lane_image):
    # Estimate pose from a single observation

    global lane_width

    if right_lane_image is None:
        # Only left lane is detected
        left_lane_ground = functions.line_image2ground(left_lane_image)
        left_slope, left_intercept = functions.line_points2slope_intercept(left_lane_ground)

        y_expected = (lane_width / 2.) - left_intercept

        left_angle = np.arctan(left_slope)
        phi_expected = -left_angle

    elif left_lane_image is None:
        # Only right lane detected
        right_lane_ground = functions.line_image2ground(right_lane_image)
        right_slope, right_intercept = functions.line_points2slope_intercept(right_lane_ground)

        y_expected = (-lane_width / 2.) + right_intercept

        right_angle = np.arctan(right_slope)
        phi_expected = -right_angle

    else:
        # Both lanes are detected
        left_lane_ground = functions.line_image2ground(left_lane_image)
        right_lane_ground = functions.line_image2ground(right_lane_image)

        # Determine Duckiebot pose from observation
        left_slope, left_intercept = functions.line_points2slope_intercept(left_lane_ground)
        right_slope, right_intercept = functions.line_points2slope_intercept(right_lane_ground)

        # Expected y position is the deviation from the centre of the left and right intercepts
        y_expected = -(left_intercept + right_intercept) / 2.

        # Convert slopes to angles
        left_angle = np.arctan(left_slope)
        right_angle = np.arctan(right_slope)

        # Expected angle is the negative of the average of the left and right slopes
        phi_expected = -((left_angle + right_angle) / 2.)

    return [y_expected, phi_expected]


class LaneFollower:
    def __init__(self):
        # CV bridge
        self.bridge = CvBridge()

        # Particle filter
        global lane_width
        self.particle_filter = ParticleFilter(1000, -lane_width / 2, lane_width / 2, -np.pi / 8., np.pi / 8.,
                                              lane_width)

        self.prev_v = 0.
        self.prev_omega = 0.
        self.prev_time = rospy.Time.now()

        self.steps_since_resample = 0

        # Publisers
        self.mask_pub = rospy.Publisher('/mask', Image, queue_size=1)
        self.edges_image_pub = rospy.Publisher('/edges_image', Image, queue_size=1)
        self.line_segments_image_pub = rospy.Publisher('/line_segments_image', Image, queue_size=1)
        self.lanes_image_pub = rospy.Publisher('/lanes_image', Image, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Image subscriber
        self.image_sub = rospy.Subscriber('/duckiebot/camera_node/image/rect', Image, self.image_callback)

    def stop(self):
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def lane_detection(self, image_bgr):
        # Convert to HSV
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Mask
        global mask_lower_threshold, mask_upper_threshold
        mask1 = cv2.inRange(image_hsv, mask_lower_threshold, mask_upper_threshold)
        # add another mask
        mask2= cv2.inRange(image_hsv, mask2_lower_threshold,mask2_upper_threshold) 
        #final mask
        mask = cv2.bitwise_or(mask1, mask2)
        # Publish mask image
        if self.mask_pub.get_num_connections() > 0:
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='passthrough'))

        # Canny edge detection
        edges = cv2.Canny(mask, 200, 400)

        # Clear the top half of the edges image
        edges[0:int(float(edges.shape[0]) / 1.5), :] = 0.

        # Publish edges image
        if self.edges_image_pub.get_num_connections() > 0:
            self.edges_image_pub.publish(self.bridge.cv2_to_imgmsg(edges, encoding='passthrough'))

        # Detect line segments
        line_segments_tmp = cv2.HoughLinesP(edges, 1, np.pi / 180., 10, None, 8, 4)

        print("line segments tmp")
        print(line_segments_tmp)
        if line_segments_tmp is None:
            print('No line segments detected')
            return [None, None]

        # Remove extra array layer from line_segments_tmp
        line_segments = []

        for line_segment in line_segments_tmp:
            line_segments.append(line_segment[0])

        # Publish line segments image
        if self.line_segments_image_pub.get_num_connections() > 0:
            line_segments_image = plot_line_segments(image_bgr, line_segments)
            self.line_segments_image_pub.publish(self.bridge.cv2_to_imgmsg(line_segments_image, encoding='bgr8'))

        # Combine line segments
        [left_lane, right_lane] = functions.average_slope_intercept(line_segments)

        if self.lanes_image_pub.get_num_connections() > 0:
            lanes_image = plot_lanes(image_bgr, left_lane, right_lane)
            self.lanes_image_pub.publish(self.bridge.cv2_to_imgmsg(lanes_image, encoding='bgr8'))

        return [left_lane, right_lane]

    def image_callback(self, image_msg):
        # Convert image message to OpenCV image
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Lane detection
        [left_lane_image, right_lane_image] = self.lane_detection(image_bgr)

        # Stop if no lanes are detected
        if left_lane_image is None or right_lane_image is None:
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return

        # Estimate position from single observation
        [y, phi] = estimate_pose(left_lane_image, right_lane_image)

        # # Estimate pose with particle filter
        # if (self.steps_since_resample >= 10):
        #     self.particle_filter.resample_particles()
        #     self.steps_since_resample = 0
        #
        # t = (rospy.Time.now() - self.prev_time).to_sec()
        #
        # [y, phi] = self.particle_filter.step(self.prev_v, self.prev_omega, t, left_lane_image, right_lane_image)

        print('y = ', y)
        print('phi = ', phi)

        # Determine control input
        v = 0.1
        omega = -1.5 * y + -2.5 * phi  # Proportional only

        print('v = ', v)
        print('omega = ', omega)

        # Publish control input
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        #cmd_vel.linear.x = 0
        #cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel)

        # Save control input for particle filter
        self.prev_v = v
        self.prev_omega = omega
        self.prev_time = rospy.Time.now()
        self.steps_since_resample += 1


if __name__ == '__main__':
    rospy.init_node('lane_follower')

    lane_follower = LaneFollower()

    rospy.on_shutdown(lane_follower.stop)

    rospy.spin()
