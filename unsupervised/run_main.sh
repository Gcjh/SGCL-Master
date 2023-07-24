for dataset in PROTEINS
do
for seed in 0
do
python main.py --seed $seed --DS $dataset
done
done
