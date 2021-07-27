PART 1:
Run the python code as :

python state_estimation_1.py filtering smoothing manhattan_error prediction most_likely_path

All the arguments are boolean

eg. if you want all inference tasks to be performed then use :
 
python state_estimation_1.py 1 1 1 1 1

All the plots will be shown on the screen

PART 2:

Run the python code as:

python state_estimation simulation estimation control sensing euclidean_error plot_ellipses plot_velocities double_object_tracking

All arguments are boolean except for control and sensing.

control can either be "normal" or "sine"
sensing can either be "normal" or "broken" (for discontinuous sensing at t=10 and 30).

eg. If you want normal control with broken sensing and want to plot uncertainty ellipses use:

python state_estimation_2.py 0 1 normal broken 0 1 0 0 

eg. simulate single object and also double object tracking(with sine control policy)

python state_estimation_2.py 1 0 sine normal 0 0 0 1