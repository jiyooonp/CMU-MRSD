#include <ros/ros.h>
#include <motion_decoder/image_converter.hpp>
#include <apriltag_ros/AprilTagDetectionArray.h>
#include <apriltag_ros/AprilTagDetection.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovariance.h>


ImageConverter* ic;
float x_det, y_det, z_det;

void apriltag_detection_callback(const apriltag_ros::AprilTagDetectionArray::ConstPtr  &msg)
{
  ROS_INFO("In subscribe\n");
  //TODO: Parse message and publish transforms as apriltag_tf and camera
  for (int i = 0; i < msg->detections.size(); i++) {
    ROS_INFO("ID: %i", msg->detections[i].id);
    geometry_msgs::PoseWithCovariance pose  = msg->detections[i].pose.pose;
    geometry_msgs::Point point = pose.pose.position;
    geometry_msgs::Quaternion quat = pose.pose.orientation;

    static tf2_ros::TransformBroadcaster br;

    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "camera";
    transformStamped.child_frame_id = "april_tf";
    transformStamped.transform.translation.x = point.x;
    transformStamped.transform.translation.y = point.y;
    transformStamped.transform.translation.z = point.z;

    tf2::Quaternion q(quat.x, quat.y, quat.z, quat.w);
    // transform.setRotation(q);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();
    br.sendTransform(transformStamped);

    x_det = point.x;
    y_det = point.y;
    z_det = point.z;
    ic->setTagLocations(x_det, y_det, z_det);
    // ROS_INFO("%f %f\n", x_det, y_det);
  }
  

  // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera", "april_tf"));
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  
  ros::NodeHandle n;
  //TODO: Add a subscriber to get the AprilTag detections The callback function skelton is given.
  ros::Subscriber sub = n.subscribe("/tag_detections", 1000, apriltag_detection_callback);
  
  ImageConverter converter;
  ic = &converter;

  ros::Rate loop_rate(50);
  ROS_INFO("In main\n");
  while(ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
