for i in {1..3}; do
  python baseline_train.py
done

for i in {1..3}; do
  python baseline_train.py --loss focal
done

python baseline_test.py