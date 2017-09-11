#!/usr/bin/env bash

export modelType="LSTM"
export batch_size=256
export epoch=100
export filename="test.csv"
#export window_size=24
export cv_folds=0
export experiment_type="train_test" # options: train_test, CVtrain_test, train_validate_test

# DANGER ZONE #
export NewFile=1


if [ $NewFile==1 ]
then
    rm $filename
    printf '%s\n' ModelType WindowSize Epochs Nodes AUC F1 Precision Recall Accuracy LabelProp | paste -sd ',' >> $filename
fi

echo $" "
echo $"--------------------------"
echo $"number of epochs:" $epoch
echo $"filename:" $filename
echo $"--------------------------"

for numNodes in "5" "10" "15" "20"  "25" "50" "75" "5 2" "10 5" "20 5" "50 15" "75 10"; do
#for numNodes in "100" "150" "200" "50 20" "75 20" "100 25"; do
for window_size in 24; do
echo $" "
echo $"----------------"
echo $"model type:" $modelType
echo $"batch size:" $batch_size
echo $"number of nodes:" $numNodes
echo $"Window size:" $window_size
echo $"----------------"
echo $" "

python3 -W ignore  main.py -model ${modelType} -bs ${batch_size} -epoch ${epoch} -nodes ${numNodes} -file ${filename} -ws ${window_size} -cv ${cv_folds} -exp ${experiment_type}
done
done
