# Multi Task Learning
Multi-Task Deep Neural Networks（MT-DNN）を使用したマルチタスク学習のコードである．<br>
詳しくはhttps://aclanthology.org/P19-1441/ を参照．<br>
本ページのコードでは日本語MC-TACOを用いてマルチタスク学習を行う．
<br>

## 使用手順
言語モデルとして xlm-roberta-large を使用し，5分割交差検証する場合を例に示す．
<br>
1. データセットとコードをダウンロードする． <br>
 `> git clone https://github.com/hi-kari2/mtl.git`
 <br>
2. 環境を構築する．<br>
 `> pip install -r requirements.txt` <br>
 `> docker pull allenlao/pytorch-mt-dnn:v1.3` <br>
 `> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v1.3 bash` <br>

 <br>
3. 用意した5分割されたデータセットをMT-DNNのフォーマットにそれぞれ変える． <br> 

 `> sh prepro.sh` 
 <br> 
中身は以下のコマンドをデータセット分用意したものである．<br>
 `> python prepro_std.py --model xlm-roberta-large --root dataset/mcduration/0/ --task_def experiments/mcduration_task_def.yml` 
<br>
4. マルチタスク学習をする<br>
 複数のデータでマルチタスク学習をする場合，使用するデータセットは同じフォルダ内に存在する必要があるため，事前に移動させておく．<br>
 `> sh run.sh` 
 <br>
 <br>
中身は5分割交差検証のためにシェルスクリプトをまとめたもので，さらにその中身はパラメータなどの設定と以下のコマンドのようになっている．<br>
 `> python train.py  --epochs 10  --task_def experiments/mcall_task_def.yml --encoder_type 5  --data_dir ${DATA_DIR} --multi_gpu_on  --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr}` 

