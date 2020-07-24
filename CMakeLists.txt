cmake_minimum_required(VERSION 2.8.3)
project(mmsegmentation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  image_transport
)

catkin_package()
