#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <sstream>

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

public:
  
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/usb_cam/image_raw", 1,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void setTagLocations(float x_det, float y_det, float z_det)
  {
	  //TODO: Update tag locations
    // float x_scaled = (x_det+1)/2 * 640;
    // float y_scaled = (y_det+1)/2 * 480;

    float x_scaled = (x_det/z_det) * 640 + 320;
    float y_scaled = (y_det/z_det) * 480 + 240;

    x_arr.push_back(x_scaled);
    y_arr.push_back(y_scaled);

  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

	//TODO: Draw circles at tag locations on image. 
      ROS_INFO("this is what : %i\n", x_arr.size());

    for(int i = 0; i < x_arr.size() ;i++)
    {
      if (i==x_arr.size()-1){
        cv::circle(cv_ptr->image, cv::Point(x_arr[i], y_arr[i]), 10, cv::Scalar(110,223,68), -1);
      }
      else{
        cv::circle(cv_ptr->image, cv::Point(x_arr[i], y_arr[i]), 10, cv::Scalar(13,107,13), -1);
      }
      // cv::circle(cv_ptr->image, cv::Point(x_arr[i], y_arr[i]), 10, cv::Scalar(13,107,13), -1);
      // std::ostringstream oss;
      // oss << x_arr[i]/688*2-1 << ", " << y_arr[i]/252*2-1;
      // std::string s = oss.str();
      // cv::putText(cv_ptr->image, s, cv::Point(x_arr[i], y_arr[i]), cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0,0,255), 0.2);
    }
    // int last = x_arr.size()-1;
    // cv::circle(cv_ptr->image, cv::Point(x_arr[last-1], y_arr[last-1]), 10, cv::Scalar(110,223,68), -1);
    
    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

    // TODO:Output modified video stream
    // Convert the modified frames into sensor_msgs::Image message and publish it using image_pub
    image_pub_.publish(cv_ptr->toImageMsg());
  }

private:
  float x_loc ,y_loc;
  std::vector<float> x_arr;
  std::vector<float> y_arr;
};
