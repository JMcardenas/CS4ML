#!/bin/bash

python --version

echo "CONTROL: starting runs on test datasets"

declare -r min_trials=1
declare -r max_trials=20

declare -r run_ID="test_run_trials_image_data_128"
declare -r example="braingen"
declare -r optimizer="Adam"
declare -r precision="single"
declare -r lrn_rate_schedule="exp_decay"
declare -r max_samples_K_tilde="1000"
declare -r weighted_LS=0

if [ "$precision" = "single" ]; then
    declare -r nb_epochs="2000"
    declare -r error_tol="5e-7"
elif [ "$precision" = "double" ]; then
    declare -r nb_epochs="5000"
    declare -r error_tol="5e-16"
fi

echo $nb_epochs

declare -A samp_percs
samp_percs[0]="0.00125"
samp_percs[1]="0.00250"
samp_percs[2]="0.00375"
samp_percs[3]="0.00500"
samp_percs[4]="0.00625"
samp_percs[5]="0.00750"
samp_percs[6]="0.00875"
samp_percs[7]="0.01000"
samp_percs[8]="0.01125"
samp_percs[9]="0.01250"
samp_percs[10]="0.01375"
samp_percs[11]="0.01500"
samp_percs[12]="0.01625"
samp_percs[13]="0.01750"
samp_percs[14]="0.01875"
samp_percs[15]="0.02000"

# for each sampling percentage
for i in {0..15..1}
do
    # for each method
    for j in {1..4..1}
    do
        echo "CONTROL: ${samp_percs[$i]} percent trials method $j"
        for (( k=$min_trials; k<=$max_trials; k++ )) 
        do
            echo "CONTROL: running trial $k of ${samp_percs[$i]} percent method $j trials"

            # grow with the arch_nodes arch_layers sizes above
            python generative_cs_example.py --nb_epochs $nb_epochs --example $example --optimizer $optimizer --quiet 0 --trial_num $k --precision $precision --run_ID $run_ID --error_tol $error_tol --lrn_rate_schedule $lrn_rate_schedule --samp_perc ${samp_percs[$i]} --samp_method $j --max_samples_K_tilde $max_samples_K_tilde --weighted_LS $weighted_LS
        done

    done
done

