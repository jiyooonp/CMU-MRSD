#include <ros/ros.h>
#include <chatbot_node/reply_msg.h>
#include <message_ui/sent_msg.h>
#include <counter_node/counter.h>
#include <arithmetic_node/arithmetic_reply.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <string>

using namespace std;

std_msgs::Header header;
string oper_type;
_Float32 answer;
_Float64 time_received;
_Float64 time_answered;
_Float64 process_time;

ros::Subscriber sent_msg_sub;

void sent_msg_callback(const message_ui::sent_msg::ConstPtr& msg){
  ROS_INFO("In the subscriber function");
  string message = msg->message;
  
  string add ="+";
  string sub ="-";
  string mult ="*";
  string div ="/";
  time_received = msg->header.stamp.toSec();
  float a =0;
  float b = 0;
  int location = 0;

  if (message.find(add) != string::npos){
    ROS_INFO("ADD");
    oper_type = "Add";
    location = message.find(add);
    a = stof(message.substr(0, location));
    b = stof(message.substr(location+1, message.length() - (location+1)));
    answer = a+b;
  }
  else if (message.find(sub) != string::npos){
    ROS_INFO("SUB");
    oper_type = "Subtract";
    location = message.find(sub);
    a = stof(message.substr(0, location));
    b = stof(message.substr(location+1, message.length() - (location+1) ));
    answer = a-b;
  }
  else if (message.find(mult) != string::npos){
    ROS_INFO("MULT");
    oper_type = "Multiply";
    location = message.find(mult);
    a = stof(message.substr(0, location));
    b = stof(message.substr(location+1, message.length() - (location+1) ));
    answer = a*b;
  }
  else if (message.find(div) != string::npos){
    ROS_INFO("DIV");
    oper_type = "Divide";
    location = message.find(div);
    a = stof(message.substr(0, location));
    b = stof(message.substr(location+1, message.length() - (location+1) ));
    answer = a/b;
  }
  else{
    ROS_INFO("IDK");
  }  
  ROS_INFO_STREAM(oper_type<<message.substr(0, location)<<":"<<
  message.substr(location+1, message.length() - (location+1)));
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "arithmetic_node");
  ros::NodeHandle n;

  ros::Publisher chatter_pub = n.advertise<arithmetic_node::arithmetic_reply>("arithmetic_reply", 20);

  sent_msg_sub = n.subscribe("sent_msg", 20, sent_msg_callback);

  ros::Rate loop_rate(20);

  while(ros::ok()) {

  time_answered = ros::Time::now().toSec();
  process_time = time_answered - time_received;
  arithmetic_node::arithmetic_reply msg;

  msg.answer= answer;
  msg.oper_type = oper_type;
  msg.header.stamp = ros::Time::now();
  msg.time_received= time_received;
  msg.time_answered= time_answered;
  msg.process_time= process_time;
  
  chatter_pub.publish(msg);

  ros::spinOnce();
  loop_rate.sleep();
  }

  return 0;
}