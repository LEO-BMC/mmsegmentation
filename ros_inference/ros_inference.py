#!/usr/bin/env python
from __future__ import print_function
import roslib
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, make_inference_return_colored_bgr
from mmseg.core.evaluation import get_palette

import time


class semantic_segmentation:
    def __init__(self):
        config_path = rospy.get_param('~config_path')
        checkpoint_path = rospy.get_param('~checkpoint_path')
        topic_sub = rospy.get_param('~topic_sub')
        topic_pub = rospy.get_param('~topic_pub')
        device = rospy.get_param('~device')
        self.palette = rospy.get_param('~palette')

        self.model = init_segmentor(config_path, checkpoint_path, device=device)

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(topic_pub, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(topic_sub, Image, self.callback, queue_size=1)


    def callback(self, data):
        print("Segmentation Callback is called")
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

        t1 = time.monotonic()
        result = inference_segmentor(self.model, image)
        t2 = time.monotonic()
        print('Segmentation took: {0}', (t2 - t1))

        image_result = make_inference_return_colored_bgr(self.model, image, result, get_palette(self.palette))

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_result, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node("semantic_segmentation", anonymous=True)
    segmentor = semantic_segmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
