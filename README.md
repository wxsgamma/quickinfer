# Quickinfer

基于 nano-vLLM 迁入的轻量 Qwen3 推理引擎，采用 quickinfer 原有的
Python 3.12、`src/quickinfer` 布局、`uv_build` 和 `quickinfer` 命令入口。

支持离线批量生成、分块 prefill、KV block 管理、前缀缓存、Tensor Parallel、
torch.compile 和 CUDA Graph。当前模型实现面向 dense Qwen3 的 FP16/BF16 safetensors 权重。
FP8、FP32 attention、MoE、多模态和其他模型架构尚未支持。

## 环境与依赖

本部署针对 Linux x86_64、NVIDIA CUDA GPU、CPython 3.12。
项目使用独立 `.venv`，所有依赖经 `pyproject.toml` 和 `uv.lock` 管理。

```bash
cd /home/hujiacong/whh/quickinfer
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run --locked quickinfer --help
```

PyTorch 固定 2.8.0 / CUDA 12.8，Triton 固定 3.4.0，Transformers 保持 4.x，
FlashAttention 固定 2.8.3 并通过 uv source 选择匹配的
[官方 wheel](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3)。
依赖和安装方式遵循 [uv 项目配置](https://docs.astral.sh/uv/concepts/projects/config/)。
如果升级 PyTorch 或 Python，需要一起更新 FlashAttention wheel 与 lock；
当前 wheel source 不适用于其他 Python 版本或操作系统。

## 离线生成

H100-3 上已有可用模型目录如下。先选择有空余显存的 GPU；例子使用 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked quickinfer generate \
  --model /data/hujiacong/whh/qwen3-1.7B \
  --prompt "What is 2 plus 2? Answer briefly." --chat \
  --max-tokens 32 --max-model-len 1024 \
  --max-num-seqs 3 --max-num-batched-tokens 128 \
  --num-kvcache-blocks 16 --gpu-memory-utilization 0.15
```

`--prompt` 可重复传入以组成 batch。`--chat` 使用 Qwen3 模板并关闭 thinking。
增加 `--enforce-eager` 可禁用 CUDA Graph；默认使用 CUDA Graph。
`example.py` 是同一 CLI 的示例包装，支持上述引擎参数。

Python API：

```python
from quickinfer import LLM, SamplingParams

with LLM(
    "/data/hujiacong/whh/qwen3-1.7B",
    max_model_len=1024,
    max_num_seqs=3,
    max_num_batched_tokens=128,
    num_kvcache_blocks=16,
    gpu_memory_utilization=0.15,
) as llm:
    outputs = llm.generate(["The capital of France is"], SamplingParams(
        temperature=0.6, max_tokens=32,
    ))
    print(outputs[0]["text"])
```

返回值为与输入顺序一致的字典列表，每项含 `text` 和 `token_ids`。
继承源实现的温度采样，不支持 temperature=0 的 greedy 模式。
输出保留原始特殊 token；按需要使用 tokenizer 解码。
一个进程同时使用一个引擎；不并发调用其 generate/step。
使用多卡 TP 时，Python 程序入口应置于 `if __name__ == "__main__":` 内，
并以 `CUDA_VISIBLE_DEVICES` 选择设备、设置 `tensor_parallel_size`。

## HTTP 服务

保留原来的 `serve` 子命令，提供本地串行、非流式 JSON 接口：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked quickinfer serve \
  --model /data/hujiacong/whh/qwen3-1.7B \
  --host 127.0.0.1 --port 8080 \
  --max-model-len 1024 --max-num-seqs 3 --max-num-batched-tokens 128 \
  --num-kvcache-blocks 16 --gpu-memory-utilization 0.15

curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompts":["The capital of France is"],"temperature":0.6,"max_tokens":32}'
```

`/generate` 接受文本或 token-ID 列表；文本直接送入模型，不自动添加聊天模板。
响应为 `{"outputs":[{"text":"...","token_ids":[...]}]}`。
这是用于研究和本地调试的简单接口，未实现 OpenAI 协议、流式输出、鉴权或并发服务。
按 Ctrl+C 退出并释放资源。

## Benchmark 与检查

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked quickinfer benchmark \
  --model /data/hujiacong/whh/qwen3-1.7B \
  --num-seqs 3 --input-len 32 --output-len 8 \
  --max-model-len 1024 --max-num-seqs 3 --max-num-batched-tokens 128 \
  --num-kvcache-blocks 16 --gpu-memory-utilization 0.15

uv run --locked python -m unittest discover -s tests -v
uv build
CUDA_VISIBLE_DEVICES=0 uv run --locked python tests/gpu_smoke.py \
  --model /data/hujiacong/whh/qwen3-1.7B
```

Benchmark 使用固定长度的随机 token 输入，报告实际生成 token 数、耗时和输出吞吐，
不包含模型加载；短测试仍可能包含特定 shape 的首次编译。
上面的参数用于检查命令能否运行，不能代表正式性能。

详细内容见 [架构与迁移映射](docs/architecture.md)、
[验证记录与边界](docs/validation.md)、[迁移计划](docs/migration-plan.md)。
源代码版权和 MIT 许可保留于 LICENSE、NOTICE。
