<!--
 * @Author: your name
 * @Date: 2021-03-17 18:27:25
 * @LastEditTime: 2021-05-16 07:51:04
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /AttentionBasedNameDisambiguation/README.md
-->

# Data

Please download [data][https://static.aminer.cn/misc/na-data-kdd18.zip] here. Unzip the file and put the data directory into project directory.

# ENV

```
conda creat -n test python==3.6.9

pip install -r requirements.txt

mkdir -p pic/FINALResult/

```


# RUN
```
python3 scripts/preprocessing.py

# prepare data
python3 HeterogeneousGraph/gen_train_data.py

# HAN Run
# localtraing
python3 HeterogeneousGraph/localHANMetricLearning.py

# DisambiguateRateSample
python3 DisambiguateRateSample/GenerateData.py
python3 DisambiguateRateSample/DisambiguateMetricLearning.py

# prepare local data
python3 HeterogeneousGraph/prepare_local_data.py

# train
python3 local/gae/train.py

```













