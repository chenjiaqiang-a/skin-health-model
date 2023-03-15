# Skin Health Model
毕业设计 *《面向脸部痤疮数据的智能严重程度分级系统
》* 的模型与实验部分的代码仓库。
## Environment
## File List
## Dataset
## Experiments
## Tips
检查GPU信息
```shell script
nvidia-smi
```
在后台运行代码
```shell script
# convert .ipynb to .py
jupyter nbconvert --to script test.ipynb
# determine gpu
CUDA_VISIBLE_DEVICES=1
# run .py script with nohup
nohup python test.py &
# run shell script
chmod o+x test.sh
./test.sh
nohuo ./test.sh &
# stop training
ps -aux
kill -9 id
```
压缩/解压文件
```shell script
# .zip
zip -r filename.zip filedir
unzip -d folder data.zip
# .tar.gz
tar -xzvf test.tar.gz
```
