### mobilebert training
bucket_name="squad_c"
bucket_root="gs://"${bucket_name}
train_tfs=${bucket_root}"/wiki/features/train/tf.record.*"
dev_tfs=${bucket_root}"/wiki/features/dev/tf.record.*"
output_dir=${bucket_root}"/mobilebert/pretrain"
teacher_checkpoint=${bucket_root}"/irbert/backup/model.ckpt-150000"
export PYTHONPATH=$PYTHONPATH:../ && python run_pretraining.py \
--attention_distill_factor=1 \
--bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json \
--bert_teacher_config_file=config/uncased_L-24_H-1024_B-512_A-4.json \
--beta_distill_factor=5000 \
--distill_ground_truth_ratio=0.5 \
--distill_temperature=1 \
--do_train \
--do_eval \
--first_input_file=${train_tfs} \
--first_max_seq_length=128 \
--first_num_train_steps=0 \
--first_train_batch_size=64 \
--gamma_distill_factor=5 \
--hidden_distill_factor=100 \
--init_checkpoint=${teacher_checkpoint} \
--input_file=${train_tfs} \
--dev_file=${dev_tfs} \
--layer_wise_warmup \
--learning_rate=0.0015 \
--max_predictions_per_seq=20 \
--max_seq_length=512 \
--num_distill_steps=240000 \
--num_train_steps=500000 \
--num_warmup_steps=10000 \
--optimizer=lamb \
--output_dir=${output_dir} \
--save_checkpoints_steps=5000 \
--train_batch_size=56 \
--use_einsum \
--use_summary \
--use_tpu \
--tpu_name="cx"