PYTORCH_JIT=0 python -m  mldaikon.collect_trace -p mnist_rnn/main.py -t megatron deepspeed torch 
PYTORCH_JIT=0 python -m  mldaikon.collect_trace -p mnist_hogwild/main.py -t megatron deepspeed torch 
