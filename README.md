<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (ICLR'24) TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting </b></h2>
</div>


<div align="center">

![](https://img.shields.io/github/last-commit/KimMeen/Time-LLM?color=green)
![](https://img.shields.io/github/stars/kwuking/TimeMixer?color=yellow)
![](https://img.shields.io/github/forks/kwuking/TimeMixer?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://openreview.net/pdf?id=7oLshfEIC2">Paper Page</a>]**
**[<a href="https://iclr.cc/virtual/2024/poster/19347">ICLR Video</a>]**
**[<a href="https://medium.com/towards-data-science/timemixer-exploring-the-latest-model-in-time-series-forecasting-056d9c883f46">Medium Blog</a>]**

**[<a href="https://mp.weixin.qq.com/s/d7fEnEpnyW5T8BN08XRi7g">中文解读1</a>]**
**[<a href="https://mp.weixin.qq.com/s/MsJmWfXuqh_pTYlwve6O3Q">中文解读2</a>]**
**[<a href="https://zhuanlan.zhihu.com/p/686772622">中文解读3</a>]**
**[<a href="https://mp.weixin.qq.com/s/YZ7L1hImIt-jbRT2tizyQw">中文解读4</a>]**

</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```bibtex
@inproceedings{wang2023timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y and ZHOU, JUN},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@article{wang2024timemixer++,
  title={TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis},
  author={Wang, Shiyu and Li, Jiawei and Shi, Xiaoming and Ye, Zhou and Mo, Baichuan and Lin, Wenze and Ju, Shengtong and Chu, Zhixuan and Jin, Ming},
  journal={arXiv preprint arXiv:2410.16032},
  year={2024}
}

```

# Updates

🚩 **News** (2024.10): TimeMixer series has been included in **[PyPOTS](https://pypots.com/)**. Many thanks to the PyPOTS team.

🚩 **News** (2024.10): We have newly released **[TimeMixer++](https://arxiv.org/abs/2410.16032)** as an upgraded version of TimeMixer. **TimeMixer++ is a time series pattern machine** that employs multi-scale and multi-resolution pattern extraction to **achieve 🏆state-of-the-art across 8 diverse analytical tasks!** 

🚩 **News** (2024.07): TimeMixer has been included in **[NeuralForecast](https://github.com/Nixtla/neuralforecast)**. Special thanks to the contributor @[marcopeix](https://github.com/marcopeix) and @[JQGoh](https://github.com/JQGoh)!

🚩 **News** (2024.07) TimeMixer has evolved into a **large model supporting comprehensive time series analysis, including long-term forecasting, short-term forecasting, anomaly detection, imputation, and classification**. In the future, we will further explore additional types of time series analysis tasks and strive to break through the limitations of current long-term forecasting to achieve efficient extreme-long-term time series forecasting.

🚩 **News** (2024.06) **Introduction of TimeMixer** in **[Chinese](https://mp.weixin.qq.com/s/d7fEnEpnyW5T8BN08XRi7g)** is available.

🚩 **News** (2024.05) TimeMixer has now released a **28-page full paper version on [arXiv](https://arxiv.org/abs/2405.14616)**. Furthermore, we have provided a **brief [video](https://iclr.cc/virtual/2024/poster/19347)** to facilitate your understanding of our work.

🚩 **News** (2024.05) TimeMixer currently **supports using future temporal features for prediction**. This feature has been well-received by the community members. You can now decide whether to enable this feature by using the parameter use_future_temporal_feature.

🚩 **News** (2024.03) TimeMixer has been included in [[**Time-Series-Library**]](https://github.com/thuml/Time-Series-Library) and achieve the consistent 🏆**state-of-the-art** in **long-term time and short-term series forecasting** .

🚩 **News** (2024.03) TimeMixer has added a time-series decomposition method based on DFT, as well as downsampling operation based on 1D convolution.

🚩 **News** (2024.02) TimeMixer has been accepted as **ICLR 2024 Poster**.

## Introduction
🏆 **TimeMixer**, as a fully MLP-based architecture, taking full advantage of disentangled multiscale time series, is proposed to **achieve consistent SOTA performances in both long and short-term forecasting tasks with favorable run-time efficiency**.

🌟**Observation 1: History Extraction** 

Given that seasonal and trend components exhibit significantly different characteristics in time series, and different scales of the time series reflect different properties, with seasonal characteristics being more pronounced at a fine-grained micro-scale and trend characteristics being more pronounced at a coarse macro scale, it is, therefore, necessary to decouple seasonal and trend components at different scales.

<p align="center">
<img src="./figures/motivation1.png"  alt="" align=center />
</p>

🌟**Observation 2: Future Prediction** 

Different scales exhibit complementary predictive capabilities when integrating forecasts from different scales to obtain the final prediction results.

<p align="center">
<img src="./figures/motivation2.png"  alt="" align=center />
</p>

## Overall Architecture
TimeMixer as a fully MLP-based architecture with **Past-Decomposable-Mixing (PDM)** and **Future-Multipredictor-Mixing (FMM)** blocks to take full advantage of disentangled multiscale series in both past extraction and future prediction phases. 

<p align="center">
<img src="./figures/overall.png"  alt="" align=center />
</p>

## Continual Federated TimeMixer

The TimeMixer backbone is unchanged. The opt-in continual layer adds bounded replay memory,
mean-shift drift detection, clustered averaging, and prediction-space mutual distillation.

The replay-free proposed mode is available through
`ContinualFedMixer(..., method='proposed')`. It does not construct or populate
`SpectralCoresetMemory`; instead each client keeps bounded diagonal-Gaussian
representation summaries in `DistributionMemory`. A frozen copy of the first
post-aggregation backbone provides the common representation space. Historical
teacher and representation losses use the current incoming batch, with their
weights modulated by retrieved distribution similarity. The legacy `legacy` mode
continues to use raw replay for comparison.

For the CSV smoke runner:

```text
python scripts/run_continual_csv.py --method proposed
```

Its Fourier probes are independently generated synthetic anchors, not slices of
client windows. Real-data experiments should provide an independently sourced
anchor tensor; private training windows must not be treated as public anchors.

### GSOD seasonal task pipeline

The upstream [data_fcl pipeline](./data_fcl_pipeline/README.md) is cloned into
`data_fcl_pipeline/`. It downloads and prepares NOAA GSOD station data into
chronological station-season NPZ tasks:

```text
.\scripts\prepare_gsod.ps1 `
  -StartYear 2010 -EndYear 2019 `
  -ArchiveDir "C:\Data\GSOD" `
  -RawDir .\data\global_surface\raw `
  -ProcessedDir .\data\global_surface\processed
python scripts\run_data_fcl.py --processed-dir .\data\global_surface\processed --method proposed
```

Add `-Stations 01000000001 01000000002` to restrict extraction to selected
11-digit NOAA station IDs. NOAA archives are large; choose the year range and
stations before downloading.

`load_season_rounds()` reads those NPZ files without loading test arrays into
training. Missing station-season tasks remain inactive (`None`) and are
excluded from federated aggregation; they are never replaced with zero data.
For real GSOD experiments, public-anchor transfer is disabled unless an
independently sourced `--public-anchor-npz` is supplied.

### Complete SSH/GPU workflow

From the repository root on the SSH machine:

```bash
cd TimeMixer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Run the commands on the Linux GPU host. On some managed Windows machines, importing
scikit-learn may fail with `Application Control policy has blocked this file` while loading
`_dist_metrics`; that is an environment DLL policy issue, not a TimeMixer error.

Download and split the paper's Electricity Consumption Dataset into household clients and
chronological continual-learning tasks:

```bash
python scripts/download_electricity_federated.py \
  --output ./data/electricity_federated \
  --clients 10 \
  --max-rows 0 \
  --rounds 4 \
  --seq-len 96 \
  --pred-len 96
```

Inspect the generated task manifest and client files:

```bash
python -m json.tool data/electricity_federated/manifest.json | less
head data/electricity_federated/clients/client_000.csv
find data/electricity_federated/clients -name '*.csv' | wc -l
```

The downloader is non-interactive and SSH-safe. Use `--max-rows 0` for the full archive. The
paper uses 89 complete clients, 15-minute samples, and 96 input plus 96 prediction steps.
Full timelines are recommended because early household streams can contain long zero-valued
prefixes that are not useful for comparison.

Run the ten-value component smoke test without a dataset:

```bash
python -m continual_federated.tiny_demo
```

Run the generic compact TimeMixer CSV smoke runner on any numeric CSV:

```bash
python scripts/run_continual_csv.py \
  --csv ./data/electricity_federated/clients/client_000.csv \
  --target load \
  --limit 0 \
  --clients 2 \
  --rounds 2
```

The CSV runner is a small single-series smoke path. The Electricity downloader is the
paper-aligned client preparation path above.

Run the FedMixer baseline alone:

```bash
python scripts/compare_fedmixer.py \
  --data-dir ./data/electricity_federated \
  --clients 2 \
  --rounds 4 \
  --limit 0 \
  --device cuda \
  --mode baseline \
  --output ./output/output.log
```

Run Continual FedMixer alone:

```bash
python scripts/compare_fedmixer.py \
  --data-dir ./data/electricity_federated \
  --clients 2 \
  --rounds 4 \
  --limit 0 \
  --device cuda \
  --mode continual \
  --output ./output/output.log
```

Run both variants in one command, which is the default:

```bash
python scripts/compare_fedmixer.py \
  --data-dir ./data/electricity_federated \
  --clients 2 \
  --rounds 4 \
  --limit 1000 \
  --device cuda \
  --mode both \
  --output ./output/output.log
```

The comparison runner defaults to `--mode all` and compares FedAvg, FedAvg-FCL,
FedMixer-FCL, and the replay-free distribution-memory method. `--mode baseline`,
`--mode continual`, or `--mode both` select the corresponding legacy subsets. Results
include ACE/BWT, per-round error, wall time, peak CUDA allocation when applicable, and
accounted tensor storage for replay/distribution memory, the reference encoder, and teachers.

The comparison commands above use the full prepared timeline. To prepare the complete
Electricity archive and all clients first:

```bash
python scripts/download_electricity_federated.py \
  --output ./data/electricity_federated \
  --clients 10 \
  --max-rows 0 \
  --rounds 4 \
  --seq-len 96 \
  --pred-len 96
```

Then run the full side-by-side comparison with `--limit 0`:

```bash
python scripts/compare_fedmixer.py \
  --data-dir ./data/electricity_federated \
  --clients 10 \
  --rounds 4 \
  --limit 0 \
  --device cuda \
  --mode all \
  --output ./output/full_output.log
```

Use `--mode baseline` or `--mode continual` to run one legacy variant, or `--mode both`
for the two FedMixer legacy variants. The default `all` mode compares all four methods.
Full mode uses every complete sliding window in each temporal task, batched in groups of 32.

### Standard single-client training

The following `run.py` command trains one client with TimeMixer. It is not the multi-client
FedMixer comparison. Add `--continual` to enable replay memory and drift detection:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id electricity \
  --model TimeMixer \
  --data custom \
  --root_path ./data/electricity_federated/ \
  --data_path clients/client_000.csv \
  --features S \
  --target load \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --train_epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --use_gpu True \
  --gpu 0 \
  --continual \
  --memory_size 200 \
  --memory_batch_size 16 \
  --memory_weight 0.1 \
  --drift_threshold 1.0
```

The comparison runner reports FedAvg, FedAvg-FCL, FedMixer-FCL, and proposed distribution
memory when run with its default `--mode all`. Results are appended to `output/output.log`.

For the original benchmark experiments, use the supplied scripts:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh
bash ./scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh
bash ./scripts/long_term_forecast/Traffic_script/TimeMixer_unify.sh
bash ./scripts/long_term_forecast/Solar_script/TimeMixer_unify.sh
bash ./scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh
bash ./scripts/short_term_forecast/M4/TimeMixer.sh
bash ./scripts/short_term_forecast/PEMS/TimeMixer.sh
```

Remove downloaded/generated artifacts when finished:

```bash
rm -rf data/electricity_federated data/ETT/ETTh1_tiny.csv
rm -rf checkpoints results test_results
```

### Past Decomposable Mixing 
we propose the **Past-Decomposable-Mixing (PDM)** block to mix the decomposed seasonal and trend components in multiple scales separately. 

<p align="center">
<img src="./figures/past_mixing1.png"  alt="" align=center />
</p>

Empowered by seasonal and trend mixing, PDM progressively aggregates the detailed seasonal information from fine to coarse and dive into the macroscopic trend information with prior knowledge from coarser scales, eventually achieving the multiscale mixing in past information extraction.

<p align="center">
<img src="./figures/past_mixing2.png"  alt="" align=center />
</p>

### Future Multipredictor Mixing 
Note that **Future Multipredictor Mixing (FMM)** is an ensemble of multiple predictors, where different predictors are based on past information from different scales, enabling FMM to integrate complementary forecasting capabilities of mixed multiscale series.

<p align="center">
<img src="./figures/future_mixing.png"  alt="" align=center />
</p>



## Get Started

1. Install requirements. ```pip install -r requirements.txt```
    > If you are using **Python 3.8**, please change the `sktime` version in `requirements.txt` to `0.29.1`
2. Download data. You can download all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
bash ./scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh
bash ./scripts/long_term_forecast/Traffic_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Solar_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Weather_script/TimeMixer.sh
bash ./scripts/short_term_forecast/M4/TimeMixer.sh
bash ./scripts/short_term_forecast/PEMS/TimeMixer.sh
```

## Main Results
We conduct extensive experiments to evaluate the performance and efficiency of TimeMixer, covering long-term and short-term forecasting, including 18 real-world benchmarks and 15 baselines.
**🏆 TimeMixer achieves consistent state-of-the-art performance in all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.

### Long-term Forecasting

To ensure fairness in model comparison, experiments were performed with standardized parameters, including aligned input lengths, batch sizes, and training epochs. Additionally, given that results in various studies often stem from hyperparameter optimization, we include outcomes from comprehensive parameter searches.

<p align="center">
<img src="./figures/long_results.png"  alt="" align=center />
</p>

### Short-term Forecasting: Multivariate data

<p align="center">
<img src="./figures/pems_results.png"  alt="" align=center />
</p>

###  Short-term Forecasting: Univariate data

<p align="center">
<img src="./figures/m4_results.png"  alt="" align=center />
</p>


## Model Abalations

To verify the effectiveness of each component of TimeMixer, we provide the detailed ablation study on every possible design in both Past-Decomposable-Mixing and Future-Multipredictor-Mixing blocks on all 18 experiment benchmarks （see our paper for full results 😊）.

<p align="center">
<img src="./figures/ablation.png"  alt="" align=center />
</p>

## Model Efficiency
We compare the running memory and time against the latest state-of-the-art models under the training phase, where TimeMixer consistently demonstrates favorable efficiency, in terms of both GPU memory and running time, for various series lengths (ranging from 192 to 3072), in addition to the consistent state-of-the-art performances for both long-term and short-term forecasting tasks.
**It is noteworthy that TimeMixer, as a deep model, demonstrates results close to those of full-linear models in terms of efficiency. This makes TimeMixer promising in a wide range of scenarios that require high model efficiency.**

<p align="center">
<img src="./figures/efficiency.png"  alt="" align=center />
</p>

## Further Reading
1, [**TimeDistill: Efficient Long-Term Time Series Forecasting with MLP via Cross-Architecture Distillation**](https://arxiv.org/abs/2502.15016), in *arXiv* 2025.

**Authors**: Juntong Ni, Zewen Liu, Shiyu Wang*, Ming Jin, Wei Jin

```bibtex
@article{ni2025timedistill,
  title={TimeDistill: Efficient Long-Term Time Series Forecasting with MLP via Cross-Architecture Distillation},
  author={Ni, Juntong and Liu, Zewen and Wang, Shiyu and Jin, Ming and Jin, Wei},
  journal={arXiv preprint arXiv:2502.15016},
  year={2025}
}
```

2, [**TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**](https://arxiv.org/abs/2410.16032), in *arXiv* 2024.

**Authors**: Shiyu Wang*, Jiawei Li*, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, Ming Jin

```bibtex
@article{wang2024timemixer++,
  title={TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis},
  author={Wang, Shiyu and Li, Jiawei and Shi, Xiaoming and Ye, Zhou and Mo, Baichuan and Lin, Wenze and Ju, Shengtong and Chu, Zhixuan and Jin, Ming},
  journal={arXiv preprint arXiv:2410.16032},
  year={2024}
}
```

3, [**Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**](https://arxiv.org/abs/2310.01728), in *ICLR* 2024.
[\[GitHub Repo\]](https://github.com/KimMeen/Time-LLM)

**Authors**: Ming Jin*, Shiyu Wang*, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

4, [**iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**](https://arxiv.org/abs/2310.06625), in *ICLR* 2024 Spotlight.
[\[GitHub Repo\]](https://github.com/thuml/iTransformer)

**Authors**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang*, Lintao Ma, Mingsheng Long

```bibtex
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Autoformer (https://github.com/thuml/Autoformer)

## Contact

If you have any questions or want to use the code, feel free to contact:

* Jiawei Li (jarvis-li@outlook.com or jli226@connect.hkust-gz.edu.cn)
* Kevin Liu (kevinliu2000@163.com)
