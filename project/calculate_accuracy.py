import pandas as pd
import glob, os
import numpy as np
from utils.const import TEST_ACCURACY, TEST_PREDICTION


all_accuracy_files = glob.glob(os.path.join(TEST_ACCURACY, "*.csv"))
accuracies_list = []
for filename in all_accuracy_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    accuracies_list.append(df)

all_accuracies_list = pd.concat(accuracies_list, axis=0, ignore_index=True)

df = all_accuracies_list.describe()
filename = f"fianl_accuracy_mean_std.csv"
outfile = TEST_ACCURACY / filename
df.to_csv(outfile)


all_inter_union_mean, all_inter_union_sd = [], []
for subject in range(0, 7): # We have 7 subjects
    filename = f"all_method_ec_subject_{subject}.npz" 
    outfile = TEST_PREDICTION / filename
    data = np.load(outfile)
    all_mean = data['inter_union_all_ec_mean']
    all_sd = data['inter_union_all_ec_sd']
    all_inter_union_mean.append(all_mean)
    all_inter_union_sd.append(all_sd)
df = pd.DataFrame({'Mean': all_inter_union_mean, 'SD': all_inter_union_sd})
filename = f"fianl_mean_std_inter_union_all.csv"
outfile = TEST_ACCURACY / filename
df.to_csv(outfile)