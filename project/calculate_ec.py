"""
Some code is inherited from https://stackoverflow.com/questions/71430032/how-to-compare-two-numpy-arrays-with-multiple-condition
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from warnings import filterwarnings
from numpy import ndarray
from itertools import combinations
import time

from utils.const import ECMethod, TEST_PREDICTION

def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    count = 0
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        count+=1
        # print("--------------Method Count-------------------", method, count)
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_voxel":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.abs(np.subtract(np.abs(r1), np.abs(r2)))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    return consistencies


if __name__ == "__main__":

    start = time.time()
    print("Starting time", start)

    # Collecting all residuals among folds and repetitetions for subject wise
    for subject in range(0, 7): # We have 7 subjects
        all_rep_residuals = []
        for rep in range(1, 11): # we have 10 repetations
            for k_fold in range(1, 6): # we have 5 folds
                filename = f"k_{k_fold}_rep_{rep}_bat_{subject}_test.npz" 
                outfile = TEST_PREDICTION / filename
                data = np.load(outfile)
                # data_residula = data['residual'].reshape(-1)
                data_target = data['target'].reshape(-1)
                data_predicts = data['predict'].reshape(-1)
                data_residula = data_predicts - data_target
                all_rep_residuals.append(data_residula)
        print("Shape of the all repated residuals of the same target image", np.shape(all_rep_residuals))
        filename = f"all_rep_residuals_bat_{subject}_test.npz"
        outfile = TEST_PREDICTION / filename
        np.savez(outfile, all_residual=all_rep_residuals)


    # Calculating EC using different methods for subject wise
    for subject in range(0, 7):
        # Reading all_residual_bat_1_test.npz file (dictionaly key = all_residual)
        filename = f"all_rep_residuals_bat_{subject}_test.npz"
        outfile = TEST_PREDICTION / filename
        data = np.load(outfile)
        all_rep_residuals = data["all_residual"]
        # print(np.shape(all_rep_residuals))

        for method in ["intersection_union_distance", "intersection_union_voxel", "intersection_union_all", 
                        "ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", ]:
            if method == "ratio":
                ratio_ec = regression_ec(all_rep_residuals, 'ratio')
                ratio_ec = np.array(ratio_ec).mean(axis=0)
                print(np.shape(ratio_ec), ratio_ec.dtype)
                ratio_ec = ratio_ec.reshape(128, 128, 128)
            elif method == "ratio-diff":
                ratio_diff_ec = regression_ec(all_rep_residuals, 'ratio-diff')
                ratio_diff_ec = np.array(ratio_diff_ec).mean(axis=0)
                ratio_diff_ec = ratio_diff_ec.reshape(128, 128, 128)
            elif method == "ratio-signed":
                ratio_sign_ec = regression_ec(all_rep_residuals, 'ratio-signed')
                ratio_sign_ec = np.array(ratio_sign_ec).mean(axis=0)
                ratio_sign_ec = ratio_sign_ec.reshape(128, 128, 128)
            elif method == "ratio-diff-signed":
                ratio_diff_sign_ec = regression_ec(all_rep_residuals, 'ratio-diff-signed')
                ratio_diff_sign_ec = np.array(ratio_diff_sign_ec).mean(axis=0)
                ratio_diff_sign_ec = ratio_diff_sign_ec.reshape(128, 128, 128)
            elif method == "intersection_union_voxel":
                inter_union_vox_ec = regression_ec(all_rep_residuals, 'intersection_union_voxel')
                inter_union_vox_ec = np.array(inter_union_vox_ec, dtype=np.float32)
                inter_union_vox_ec = np.array(inter_union_vox_ec).mean(axis=0)
                # print(np.shape(inter_union_vox_ec), inter_union_vox_ec.dtype)
                inter_union_vox_ec = inter_union_vox_ec.reshape(128, 128, 128)
            elif method == "intersection_union_all":
                inter_union_all_ec = regression_ec(all_rep_residuals, 'intersection_union_all')
                inter_union_all_ec = np.array(inter_union_all_ec, dtype=np.float32)
                inter_union_all_ec_mean = np.array(inter_union_all_ec).mean() #axis=1, as this is only a 1D array
                inter_union_all_ec_sd = np.array(inter_union_all_ec).std(ddof=1)
                # print("----------Mean and SD---------", inter_union_all_ec_mean, inter_union_all_ec_sd)
            elif method == "intersection_union_distance":
                inter_union_distance_ec = regression_ec(all_rep_residuals, 'intersection_union_distance')
                inter_union_distance_ec = np.array(inter_union_distance_ec, dtype=np.float32)
                inter_union_distance_ec = np.array(inter_union_distance_ec).mean(axis=0)
                inter_union_distance_ec = inter_union_distance_ec.reshape(128, 128, 128)
            

        filename = f"all_method_ec_subject_{subject}.npz"
        outfile = TEST_PREDICTION / filename
        np.savez(outfile, ratio=ratio_ec, ratio_diff=ratio_diff_ec, ratio_sign=ratio_sign_ec, ratio_diff_sign=ratio_diff_sign_ec,
                inter_union_vox=inter_union_vox_ec, inter_union_distance=inter_union_distance_ec, 
                inter_union_all_ec_mean=inter_union_all_ec_mean, inter_union_all_ec_sd=inter_union_all_ec_sd)
        duration = time.time() - start
        print("Time in seconds   ", duration)