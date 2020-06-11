#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose 
from ar_track_alvar_msgs.msg import AlvarMarkers
import numpy as np

yar=0
zar=0
xar=0

def print_arMarker(msg):

	global xar
	global yar
	global zar

	
	if (not msg.markers):
		print("not detected yet!")
	
	else:
		xar=msg.markers[0].pose.pose.position.x
		yar=msg.markers[0].pose.pose.position.y
		zar=msg.markers[0].pose.pose.position.z


		print("xar, %d", xar)
		print("yar, %d", yar)
		print("zar, %d", zar)

		dist=np.sqrt(xar**2 +yar**2 + zar**2)
		print("dist=, %d", dist)
		
		if zar < 0.15:
			print("detect state")
	
			
	

ar_sub = rospy.Subscriber('/ar_pose_marker',AlvarMarkers,print_arMarker)



rospy.init_node('get_ar_distance')



while not rospy.is_shutdown():
	rospy.spin()