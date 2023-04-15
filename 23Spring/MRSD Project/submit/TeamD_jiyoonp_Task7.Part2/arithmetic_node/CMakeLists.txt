cmake_minimum_required(VERSION 3.0.2)
project(arithmetic_node)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

add_message_files(
    FILES
    arithmetic_reply.msg
)
generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(arithmetic_node src/arithmetic_node.cpp)
add_dependencies(arithmetic_node ${catkin_EXPORTED_TARGETS})
add_dependencies(arithmetic_node message_ui_generate_messages_cpp)
target_link_libraries(arithmetic_node ${catkin_LIBRARIES} )