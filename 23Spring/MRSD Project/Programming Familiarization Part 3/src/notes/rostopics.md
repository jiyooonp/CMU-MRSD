## rosrun apriltag_ros apriltag_ros_continuous_node
    /camera_info
    /image_rect
    /rosout
    /rosout_agg
    /tag_detections
    /tf

## roslaunch apriltag_ros continuous_detection.launch
    /camera_rect/camera_info
    /camera_rect/image_rect
    /rosout
    /rosout_agg
    /tag_detections
    /tag_detections_image
    /tag_detections_image/compressed
    /tag_detections_image/compressed/parameter_descriptions
    /tag_detections_image/compressed/parameter_updates
    /tag_detections_image/compressedDepth
    /tag_detections_image/compressedDepth/parameter_descriptions
    /tag_detections_image/compressedDepth/parameter_updates
    /tag_detections_image/theora
    /tag_detections_image/theora/parameter_descriptions
    /tag_detections_image/theora/parameter_updates
    /tf

## rostopic info /tag_detections
    Type: apriltag_ros/AprilTagDetectionArray

    Publishers: 
    * /apriltag_ros_continuous_node (http://jen:42525/)

    Subscribers: None


## rosmsg show apriltag_ros/AprilTagDetectionArray 
    [message page](http://docs.ros.org/en/melodic/api/apriltag_ros/html/msg/AprilTagDetectionArray.html)
    
    std_msgs/Header header
        uint32 seq
        time stamp
        string frame_id
    apriltag_ros/AprilTagDetection[] detections
        int32[] id
        float64[] size
        geometry_msgs/PoseWithCovarianceStamped pose
            std_msgs/Header header
                uint32 seq
                time stamp
                string frame_id
            geometry_msgs/PoseWithCovariance pose
                geometry_msgs/Pose pose
                    geometry_msgs/Point position
                        float64 x
                        float64 y
                        float64 z
                    geometry_msgs/Quaternion orientation
                        float64 x
                        float64 y
                        float64 z
                        float64 w
                float64[36] covariance

## rosmsg show apriltag_ros/AprilTagDetection
    [message page](http://docs.ros.org/en/kinetic/api/apriltag_ros/html/msg/AprilTagDetection.html)

    int32[] id
    float64[] size
    geometry_msgs/PoseWithCovarianceStamped pose
        std_msgs/Header header
            uint32 seq
            time stamp
            string frame_id
        geometry_msgs/PoseWithCovariance pose
            geometry_msgs/Pose pose
                geometry_msgs/Point position
                    float64 x
                    float64 y
                    float64 z
                geometry_msgs/Quaternion orientation
                    float64 x
                    float64 y
                    float64 z
                    float64 w
            float64[36] covariance
## rostopic info /usb_cam/image_raw
    Type: sensor_msgs/Image

    Publishers: 
    * /player (http://jen:36825/)

    Subscribers: 
    * /apriltag_detector (http://jen:34593/)


## rostopic info /usb_cam/camera_info 
    Type: sensor_msgs/CameraInfo

    Publishers: 
    * /player (http://jen:36825/)

    Subscribers: 
    * /apriltag_detector (http://jen:34593/)

## rostopic info /usb_cam/image_raw
    Type: sensor_msgs/Image

    Publishers: None

    Subscribers: 
    * /apriltag_detector (http://jen:34593/)

## rosnode info /hector_trajectory_server 
    --------------------------------------------------------------------------------
    Node [/hector_trajectory_server]
    Publications: 
    * /rosout [rosgraph_msgs/Log]
    * /trajectory [nav_msgs/Path]

    Subscriptions: 
    * /syscommand [unknown type]
    * /tf [tf2_msgs/TFMessage]
    * /tf_static [unknown type]

    Services: 
    * /hector_trajectory_server/get_loggers
    * /hector_trajectory_server/set_logger_level
    * /trajectory
    * /trajectory_recovery_info


    contacting node http://jen:46037/ ...
    Pid: 39679
    Connections:
    * topic: /rosout
        * to: /rosout
        * direction: outbound (49165 - 127.0.0.1:46444) [13]
        * transport: TCPROS
    * topic: /tf
        * to: /apriltag_detector (http://jen:36237/)
        * direction: inbound (40910 - jen:44581) [12]
        * transport: TCPROS
    * topic: /tf
        * to: /motion_decoder (http://jen:37997/)
        * direction: inbound (46202 - jen:35879) [14]
        * transport: TCPROS



1. change apriltags_ros.launch 
    <launch>
    <node pkg="apriltag_ros" type="apriltag_ros_continuous_node" name="apriltag_detector" output="screen">
        <!-- Remap topic required by the node to custom topics -->
        <remap from="image_rect" to="/usb_cam/image_raw" />
        <remap from="camera_info" to="/usb_cam/camera_info" />

        <!-- Optional: Subscribe to the compressed stream-->
        <param name="image_transport" type="str" value="raw" />

        <!-- Select the tag family: 16h5, 25h7, 25h9, 36h9, or 36h11(default) -->
        <param name="tag_family" type="str" value="tag36h11" />

        <!-- Enable projected optical measurements for more accurate tag transformations -->
        <!-- This exists for backwards compatability and should be left true for new setups -->
        <param name="projected_optics" type="bool" value="true" />

        <!-- Describe the tags -->
        <rosparam param="standalone_tags">[
        {id: 0, size: 0.163513},
        {id: 1, size: 0.163513, frame_id: a_frame},
        {id: 2, size: 0.163513, frame_id: tag_2},
        {id: 3, size: 0.163513},
        {id: 4, size: 0.163513},
        {id: 5, size: 0.163513}]
        </rosparam>
    </node>
    </launch>





2. Failed to load module "canberra-gtk-module"

        sudo apt-get install libcanberra-gtk-module

3. in apriltags_ros.launch change to 

        <param name="tag_family" type="str" value="tag36h11" />

4. Change CMakefile and the cpp file: apriltags_ros to apriltag_ros

5. Error: The program 'Image window' received an X Window System error.
    update & upgrade ur apt-get / apt

6. visualization 
    run rviz
    change /Global Options/Fixed Frame to camera (was map)


7. img size 
    height: 480
    width: 640
