# Conveyor tracking dashboard

The dashboard used for this exercise is designed to connect detections of varying quality into continuous trajectories. On the input, it takes a time series data of detections (horizontal axis of the chart) and the position along the conveyor (vertical axis). The focus of the dashboard is on the matrix (Nrois x Nframes) where roi (region of interest) is scored for a presence of a material on an industrial conveyor moving in steps. The material does not have a unique appearance and therefore, the individual tracking requires consolidation techniques to average signal over different sources and different single trajectories. 

Example of how the dashboard should be used is shown on a user's screencast: exercise2/example.webm

## Task 

1. Familiarise with the dashboard and the conversions from frames to datetime and roi to physical coordinate y. 
2. Add a callback feature which can connect the trajectories on the bottom to the ends of the individual signals located in the upper half. 
3. Add a saving routine for data where you save: 
 - datetime start of the trajectory
 - datetime end of the trajectory
 - y of the end of the trajectory

# Setup 
1. Create venv with python3.8 and packages from requirements. 
2. Running the dashboard 
```
cd exercise2/
python conveyor_tracking_dashboard.py
```
