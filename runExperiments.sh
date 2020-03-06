for conv in conv2 conv4 conv6 conv8; do
    {
    for init in ukn usc; do
        for prune_rate in 0.3 0.5 0.7; do
            python main.py --config configs/smallscale/"${conv}"/"${conv}"_"${init}"_unsigned.yml --multigpu 8 --data  dataset/ --prune-rate "${prune_rate}"
        done
    done
    }&
done

for conv in conv2 conv4 conv6 conv8; do
    {
    for opt in adam sgd; do
            python main.py --config configs/smallscale_baselines/dense/"${conv}"/"${conv}"_"${opt}"_baseline.yml --multigpu 9 --data  dataset/
    done
    }&
done