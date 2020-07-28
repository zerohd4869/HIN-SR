cat run_bert_reviews_toys.sh 
export CUDA_VISIBLE_DEVICES=0
for((i=0;i<1;i++));  
do   

python run_bert_review_toys.py \
--model_type bert \
--model_name_or_path uncased_L-12_H-768_A-12 \
--do_train \
--do_eval \
--data_dir ./review_data/reviews_Toys_and_Games_5_bm25_top3_stop_0115   \
--output_dir ./model_reviews_Toys_top3_0315_v0_f1  \
--output_dir_acc ./model_reviews_Toys_top3_0315_v0_acc  \
--max_seq_length 128 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 150 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 20000 

done  





