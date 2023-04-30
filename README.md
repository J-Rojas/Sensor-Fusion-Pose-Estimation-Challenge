# Obstacle pose estimation using neural networks and lidar data

## Original Didi-Challenge scope

The original scope of work was completed in 2017 for the [Didi-Chuxing Perception Challenge](https://github.com/jae1e/Didi-Udacity-SelfDrivingCar). My team consisted of working professionals who particpated in the Udacity Autonomous Vehicles training program but who had no prior research experience with autonomous perception and neural networks. Our objectives were to analyze the dataset and build a functioning ROS node for process sensor data from lidar cameras.

## Approach

We followed some of the concepts presented in the follow paper: ["Multi-View 3D Object Detection Network for Autonmous Driving](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf). A simple fully convolutional neural network to process lidar "bird's eye view" and 360-degree range scans was designed and trained.

## Outcome

Our submission to the competition resulted in team scoring in the top 10% of the leaderboard. Our entire amateur team consisted of engineers had never worked on perception problems before so we benefited greatly from this learning experience by being able to understand how to design neural networks, cleanse data, and offer refinements to the model and training process to improve our model's performance.

## Future Direction

This code base will be updated to use PyTorch and more modern methods. Higher-quality datasets such as Waymo's Perception dataset will be targeted for use to streamline the pretraining and training processes.