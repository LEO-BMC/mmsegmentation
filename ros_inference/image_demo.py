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
        self.image_pub = rospy.Publisher("/segmentation_result/cam_fm_01/image_raw", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/spinnaker_ros_driver_node/cam_fm_01/image_raw", Image, self.callback)

        parser = ArgumentParser()
        parser.add_argument('img', help='Image file')
        parser.add_argument('config', help='Config file')
        parser.add_argument('checkpoint', help='Checkpoint file')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--palette',
            default='cityscapes',
            help='Color palette used for segmentation map')
        self.args = parser.parse_args()

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(self.args.config, self.args.checkpoint, device=self.args.device)

    def callback(self, data):
        print("Segmentation Callback is called")

        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

        t1 = time.monotonic()
        result = inference_segmentor(self.model, image)
        t2 = time.monotonic()
        print('Segmentation took: {0}',  (t2 - t1))

        # show the results
        image_result = make_inference_return_colored_bgr(self.model, image, result, get_palette(self.args.palette))

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_result, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    segmentor = semantic_segmentation()
    rospy.init_node("semantic_segmentation", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
