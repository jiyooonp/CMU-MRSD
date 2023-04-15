#include <ros/ros.h>
#include <chatbot_node/reply_msg.h>
#include <message_ui/sent_msg.h>
#include <std_msgs/Header.h>
#include <string>

using namespace std;

string response;

//Add your code here
void chatterCallback(const message_ui::sent_msg::ConstPtr& msg){
  string message = msg->message;
  string hello ="Hello";
  string whatIs = "What is your name";
  string howAre = "How";

  response = "None of the above";

  if (message.find(hello) != string::npos){
    ROS_INFO("Hello");
    response="Hello";
  }
  else if (message.find(whatIs) != string::npos){
    ROS_INFO("Name");
    response= "My name is MRSD Siri";
  }
  else if (message.find(howAre) != string::npos){
    ROS_INFO("How Are You?");
    response = "I am fine, thank you.";
  }
  else{
    ROS_INFO("IDK");
    response = "IDK what you are saying";
  }  
}


int main(int argc, char **argv) {

  ros::init(argc, argv, "chatbot_node");
  std::string name;
  ros::NodeHandle n;
  n.setParam("/name", "Jiyoon");
  n.getParam("/name", name);
  ros::Subscriber sub = n.subscribe("sent_msg", 20, chatterCallback);
  ros::Publisher chatter_pub = n.advertise<chatbot_node::reply_msg>("reply_msg", 20);
  ros::Rate loop_rate(20);

  int count = 0;
  response = "Nothing Yet";
  ROS_INFO_STREAM(response);

  while(ros::ok()) {
    

    chatbot_node::reply_msg msg;
    std_msgs::Header head;

    msg.header = head;
    if (response.find("Hello") != string::npos)
       msg.message = response + " " + name ;
    else
      msg.message = response;

    chatter_pub.publish(msg);

    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }

  return 0;
}