from pipeline import *
from viz import *

batch_dir = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/B-100-T"
data_p = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/B-100-T-Results/data"
viz_p = r"/Volumes/BBall_Data_23X_pt2/HUDL_basketball_data_pt2/22-23/B-100-T-Results/viz"

process_dir(batch_dir, data_p, viz_p)
