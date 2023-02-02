/*
 * yolo_object_detector_node.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include <rclcpp/rclcpp.hpp>
#include "darknet_ros/YoloObjectDetector.hpp"


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto yoloObjectDetector = std::make_shared<darknet_ros::YoloObjectDetector>("yolo_detector");

  yoloObjectDetector->init();
  
  rclcpp::spin(yoloObjectDetector->get_node_base_interface());

  rclcpp::shutdown();

  return 0;
}
