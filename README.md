<div align="center">

# SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/RBAjeWSA)




<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> 
  </p>
</div>

</div>


# News
- **[2025/05/06]** 🎉 We released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

# Links

- 📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)


# Getting Started
This repository contains training code for the `SkyRL-v0` release. Our implementation is a fork of [VeRL](https://github.com/volcengine/verl).   
The repo is currently utilizing the SGLang async rollout feature introduced to VeRL in this draft [PR](https://github.com/volcengine/verl/pull/917), based on this [commit](https://github.com/volcengine/verl/commits/436530d6ec2b449e035fd2e823ad5b363cbf908e). We will refactor the code soon so that the codebase can easily keep up with VeRL main branch.

## Installation

The first step is to clone our repository:

```bash 
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL
```

For detailed installation instructions, please refer to [INSTALL.md](./INSTALL.md)

## Scripts for reproduction

For reproducing our results for SkyRL-Agent-14B-v0, SkyRL-Agent-8B-v0, and SkyRL-Agent-7B-v0 you can refer to [examples/sky](./examples/sky/README.md).

# Evaluation
We report the evaluation result on SWE-Bench-Verified below:

| Model              | Base                 | Base Performance | Performance | Training Time |
|--------------------|----------------------|------------------|-------------|---------------|
| SkyRL-Agent-7B-v0  | OpenHands-7B-Agent   | 11%              | 14.6%       | 16hrs 8xH100  |
| SkyRL-Agent-8B-v0  | Qwen3-8B no thinking | 3.6%             | 9.4%        | 27hrs 8xH200  |
| SkyRL-Agent-14B-v0 | Qwen3-14B thinking   | 18%              | 21.6%       | 20hrs 8xH200  |


# Acknowledgement
This work is done at [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/), with the amazing compute support from [Anyscale](https://www.anyscale.com/), [Databricks](https://www.databricks.com/), [NVIDIA](https://developer.nvidia.com/brev), [Lambda Labs](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [AMD](https://www.amd.com/).

We would also like to thank [Ying Sheng](https://sites.google.com/view/yingsheng/home), [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/) from [SGLang](https://github.com/sgl-project/sglang) team for supporting SGLang async rollout integration, and [Kaichao You](https://youkaichao.github.io/research), [Simon Mo](https://github.com/simon-mo) from [vLLM](https://github.com/vllm-project/vllm) team for supporting vLLM performance optimization.

# Citation
The code in this repository is mostly described in the post below. Please consider citing this work if you find the repository helpful. 

```bibtex
@misc{cao2025skyrl,
  title     = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author    = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and Shu Liu and Eric Tang and Jiayi Pan and Xingyao Wang and Akshay Malik and Graham Neubig and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
  year      = {2025},
}
```
