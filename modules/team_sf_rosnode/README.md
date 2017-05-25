### Usage
This package requires ROS on Ubuntu 16.04 (or Docker container running on Mac with Quartz).

On your Ubuntu machine (container), install following:

```
apt-get -y install python-pip
pip install --upgrade pip
pip install numpy
apt install wget
apt update && apt upgrade
bash <(wget -q -O - https://bitbucket.org/DataspeedInc/ros_binaries/raw/default/scripts/setup.bash)
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FF6D3CDA
sudo sh -c 'echo "deb [ arch=amd64 ] http://packages.dataspeedinc.com/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-dataspeed-public.list'
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FF6D3CDA
sh -c 'echo "deb [ arch=amd64 ] http://packages.dataspeedinc.com/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-dataspeed-public.list'
apt-get update
sh -c 'echo "yaml http://packages.dataspeedinc.com/ros/ros-public-'$ROS_DISTRO'.yaml '$ROS_DISTRO'" > /etc/ros/rosdep/sources.list.d/30-dataspeed-public-'$ROS_DISTRO'.list'
rosdep update
apt-get install ros-$ROS_DISTRO-dbw-mkz
apt-get install ros-$ROS_DISTRO-mobility-base
apt-get install ros-$ROS_DISTRO-baxter-sdk
apt-get install ros-$ROS_DISTRO-velodyne
apt-get update && sudo apt-get upgrade && rosdep update
```

Make the node executable
```
chmod +x <Path to team_sf_rosnode folder>/scripts/lidar_predict.py
```

Add the package into your env
```
export ROS_PACKAGE_PATH=<Path to team_sf_rosnode folder>:$ROS_PACKAGE_PATH
```

Launch the visualiver:
```
roslaunch team_sf_rosnode visualize.launch
```

Visualiz a bag file:
```
roslaunch team_sf_rosnode lidar_predict.launch bag:=<BAG_PATH>
```
