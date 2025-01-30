import subprocess
import re

seq_len = 336
model_name = 'DLinear'

argv = (
    '--is_training 0 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_{seq_len}_{pred_len} \
    --model {model_name} \
    --data custom \
    --features M \
    --seq_len {seq_len} \
    --pred_len {pred_len} \
    --enc_in 8 \
    --des Exp \
    --itr 1 --batch_size {batch_size} --learning_rate 0.0005'
)

pred_len = 96
batch_size = 8

f = open(f'logs/LongForecasting/{model_name}_Exchange_{seq_len}_{pred_len}.log', 'w')
subprocess.call(['python3', 'run_longExp.py', *re.split('\s+', argv.format(
    seq_len=seq_len, pred_len=pred_len, model_name=model_name, batch_size=batch_size
))], stdout=f)
