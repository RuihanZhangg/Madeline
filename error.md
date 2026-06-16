(madeline) zhangruihan@SYS-2029GP-TR-01:~/Madeline$ bash experiments/scripts/run_baseline.sh large 50
======================================
Running BASELINE ZeRO-3 (no caching)
  Model: GPT-2 large
  Steps: 50
  GPUs:  2
======================================
[2026-06-16 23:41:40,339] [WARNING] [runner.py:232:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-06-16 23:41:40,340] [INFO] [runner.py:630:main] cmd = /home/zhangruihan/miniconda3/envs/madeline/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMV19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None --log_level=info /home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py --model_size large --num_steps 50 --deepspeed_config /home/zhangruihan/Madeline/experiments/scripts/../configs/ds_config_baseline.json
[2026-06-16 23:41:44,902] [INFO] [launch.py:162:main] WORLD INFO DICT: {'localhost': [0, 1]}
[2026-06-16 23:41:44,902] [INFO] [launch.py:168:main] nnodes=1, num_local_procs=2, node_rank=0
[2026-06-16 23:41:44,902] [INFO] [launch.py:179:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1]})
[2026-06-16 23:41:44,902] [INFO] [launch.py:180:main] dist_world_size=2
[2026-06-16 23:41:44,902] [INFO] [launch.py:184:main] Setting CUDA_VISIBLE_DEVICES=0,1
[2026-06-16 23:41:44,903] [INFO] [launch.py:272:main] process 1284970 spawned with command: ['/home/zhangruihan/miniconda3/envs/madeline/bin/python3.10', '-u', '/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py', '--local_rank=0', '--model_size', 'large', '--num_steps', '50', '--deepspeed_config', '/home/zhangruihan/Madeline/experiments/scripts/../configs/ds_config_baseline.json']
[2026-06-16 23:41:44,904] [INFO] [launch.py:272:main] process 1284971 spawned with command: ['/home/zhangruihan/miniconda3/envs/madeline/bin/python3.10', '-u', '/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py', '--local_rank=1', '--model_size', 'large', '--num_steps', '50', '--deepspeed_config', '/home/zhangruihan/Madeline/experiments/scripts/../configs/ds_config_baseline.json']
Model: GPT-2 large, Parameters: 774,030,080
Model: GPT-2 large, Parameters: 774,030,080
/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/utils/cpp_extension.py:361: UserWarning: 

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) is not compatible with the compiler Pytorch was
built with for this platform, which is g++ on linux. Please
use g++ to to compile your extension. Alternatively, you may
compile PyTorch from source using c++, and then you can also use
c++ to compile your extension.

See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
with compiling PyTorch from source.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(WRONG_COMPILER_WARNING.format(
/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/utils/cpp_extension.py:1964: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
[2026-06-16 23:42:02,990] [WARNING] [engine.py:1690:_configure_basic_optimizer] FusedAdam CUDA build failed (Error building extension 'fused_adam': [1/3] c++ -MMD -MF fused_adam_frontend.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/includes -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/TH -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/zhangruihan/miniconda3/envs/madeline/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -UC10_USE_GLOG -c /home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam/fused_adam_frontend.cpp -o fused_adam_frontend.o 
FAILED: [code=127] fused_adam_frontend.o 
c++ -MMD -MF fused_adam_frontend.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/includes -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/TH -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/zhangruihan/miniconda3/envs/madeline/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -UC10_USE_GLOG -c /home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam/fused_adam_frontend.cpp -o fused_adam_frontend.o 
/bin/sh: 1: c++: not found
[2/3] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output multi_tensor_adam.cuda.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/includes -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/TH -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/zhangruihan/miniconda3/envs/madeline/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -O3 -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -lineinfo --use_fast_math -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -UC10_USE_GLOG -std=c++17 -c /home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam/multi_tensor_adam.cu -o multi_tensor_adam.cuda.o 
FAILED: [code=1] multi_tensor_adam.cuda.o 
/usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output multi_tensor_adam.cuda.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/includes -I/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/TH -isystem /home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /home/zhangruihan/miniconda3/envs/madeline/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -O3 -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -lineinfo --use_fast_math -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -UC10_USE_GLOG -std=c++17 -c /home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/ops/csrc/adam/multi_tensor_adam.cu -o multi_tensor_adam.cuda.o 
gcc: fatal error: cannot execute ‘cc1plus’: execvp: 没有那个文件或目录
compilation terminated.
nvcc fatal   : Failed to preprocess host compiler properties.
ninja: build stopped: subcommand failed.
), falling back to torch.optim.AdamW / torch.optim.Adam.
[2026-06-16 23:42:03,073] [WARNING] [engine.py:1690:_configure_basic_optimizer] FusedAdam CUDA build failed (/home/zhangruihan/.cache/torch_extensions/py310_cu121/fused_adam/fused_adam.so: cannot open shared object file: No such file or directory), falling back to torch.optim.AdamW / torch.optim.Adam.
[Rank 0] time (ms) | init_optimizer_state: 0.00
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
[Rank 0] time (ms) | fwd_microstep: 1057.35 | bwd_microstep: 563.43 | bwd_inner_microstep: 497.01 | bwd_allreduce_microstep: 66.16 | step_microstep: 0.05
Step    0 | Loss: 11.0838 | Time: 1.626s | Tokens/s: 630 | Fwd+Bwd Mem: 7.89 GB | Step Mem: 7.89 GB
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py", line 180, in <module>
[rank1]:     main()
[rank1]:   File "/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py", line 135, in main
[rank1]:     model_engine.step()
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/engine.py", line 2822, in step
[rank1]:     self._take_model_step(lr_kwargs)
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/engine.py", line 2717, in _take_model_step
[rank1]:     self.optimizer.step()
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/utils/nvtx.py", line 20, in wrapped_fn
[rank1]:     ret_val = func(*args, **kwargs)
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 2464, in step
[rank1]:     self._optimizer_step(sub_group_id)
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 1111, in _optimizer_step
[rank1]:     step_with_gradscaler(self.optimizer)
[rank1]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 1095, in step_with_gradscaler
[rank1]:     optimizer.step()
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 487, in wrapper
[rank1]:     out = func(*args, **kwargs)
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 91, in _use_grad
[rank1]:     ret = func(self, *args, **kwargs)
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 223, in step
[rank1]:     adam(
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 154, in maybe_fallback
[rank1]:     return func(*args, **kwargs)
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 784, in adam
[rank1]:     func(
[rank1]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 611, in _multi_tensor_adam
[rank1]:     exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.44 GiB. GPU 1 has a total capacity of 10.75 GiB of which 748.75 MiB is free. Including non-PyTorch memory, this process has 10.01 GiB memory in use. Of the allocated memory 8.43 GiB is allocated by PyTorch, and 1.32 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py", line 180, in <module>
[rank0]:     main()
[rank0]:   File "/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py", line 135, in main
[rank0]:     model_engine.step()
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/engine.py", line 2822, in step
[rank0]:     self._take_model_step(lr_kwargs)
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/engine.py", line 2717, in _take_model_step
[rank0]:     self.optimizer.step()
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/utils/nvtx.py", line 20, in wrapped_fn
[rank0]:     ret_val = func(*args, **kwargs)
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 2464, in step
[rank0]:     self._optimizer_step(sub_group_id)
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 1111, in _optimizer_step
[rank0]:     step_with_gradscaler(self.optimizer)
[rank0]:   File "/home/zhangruihan/Madeline/_deepspeed_ref/deepspeed/runtime/zero/stage3.py", line 1095, in step_with_gradscaler
[rank0]:     optimizer.step()
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 487, in wrapper
[rank0]:     out = func(*args, **kwargs)
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 91, in _use_grad
[rank0]:     ret = func(self, *args, **kwargs)
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 223, in step
[rank0]:     adam(
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/optimizer.py", line 154, in maybe_fallback
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 784, in adam
[rank0]:     func(
[rank0]:   File "/home/zhangruihan/miniconda3/envs/madeline/lib/python3.10/site-packages/torch/optim/adam.py", line 611, in _multi_tensor_adam
[rank0]:     exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.44 GiB. GPU 0 has a total capacity of 10.75 GiB of which 746.25 MiB is free. Including non-PyTorch memory, this process has 10.01 GiB memory in use. Of the allocated memory 8.43 GiB is allocated by PyTorch, and 1.32 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[2026-06-16 23:42:11,933] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 1284970
[2026-06-16 23:42:12,076] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 1284971
[2026-06-16 23:42:12,076] [ERROR] [launch.py:341:sigkill_handler] ['/home/zhangruihan/miniconda3/envs/madeline/bin/python3.10', '-u', '/home/zhangruihan/Madeline/experiments/scripts/train_gpt2.py', '--local_rank=1', '--model_size', 'large', '--num_steps', '50', '--deepspeed_config', '/home/zhangruihan/Madeline/experiments/scripts/../configs/ds_config_baseline.json'] exits with return code = 1