#!/usr/bin/env bash

export modelType="LSTM"
export batch_size=256
export epoch=100
export filename="test.csv"
export window_size=24
export cv_folds=0
export experiment_type="train_test" # options: train_test, CVtrain_test, train_validate_test
# DANGER ZONE #
# If NewFile=1, the old filename is removed and a new file is created
export NewFile=0


if [ $NewFile=1 ]
then
    rm $filename
    printf '%s\n' ModelType Epochs Nodes AUC F1 Precision Recall Accuracy | paste -sd ',' >> $filename
fi

echo $" "
echo $"---------------------"
echo $"number of epochs:" $epoch
echo $"filename:" $filename
echo $"---------------------"

for numNodes in "100"; do
echo $" "
echo $"--------------------"
echo $"model type:" $modelType
echo $"batch size:" $batch_size
echo $"number of nodes:" $numNodes
echo $"--------------------"
echo $" "
python3 main.py -model ${modelType} -bs ${batch_size} -epoch ${epoch} -nodes ${numNodes} -f ${filename} -ws ${window_size} -cv ${cv_folds} -exp ${experiment_type}
done