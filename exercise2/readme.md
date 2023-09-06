# Conveyor tracking dashboard

The following dashboard is designed to track detections of varying quality. It is a time series data of detections as a cross section of some physical space. The main focus of the dashboard is on the matrix (Nrois x Nframes) where roi (region of interest) is scored for a presence of a material on an industrial conveyor. The material does not have a unique appearance and therefore the individual tracking requires, consolidation techniques to average signal over different sources and different single trajectories. 

Example of how the dashboard should look like: exercise2/example.webm

## Task 

1. Familiriase with the dashboard and the conversions from frames to datetime and roi to physical coordinate y. 
2. Add a callback feature which can connect the trajectories on the bottom to the ends of the individual signals located in the upper half. 
3. Add a saving routine for data where you save: 
 - datetime start of the trajectory
 - datetime end of the trajectory
 - y of the end of the trajectory

# Setup 
1. Create venv with python3.8 and packages from requirements. 
2. Running the dashboard 
