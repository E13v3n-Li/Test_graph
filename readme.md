## Traceable Black-Box Watermarks for Federated Learning


This is the official implementation for the ICLR'26 paper "Traceable Black-Box Watermarks for Federated Learning". You can find the full version of our paper [here][paper]. 

[paper]: https://arxiv.org/pdf/2505.13651

If you have any issues using this repo, feel free to contact Jiahao @ jiahaox@unr.edu.

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `tramark.yaml` for your convenience.

### Dataset

All datasets are hosted on `torchvision` and download automatically, with the exception of Tiny-ImageNet, which is available via Kaggle.

### Example

Generally, to run an experiment by using our TraMark, you can easily use the following command:

```
python federated.py \
--watermarking_method tramark \
--dataset cifar10 --num_clients 10 \
--alpha 0.04 \
--k 0.01 \
--local_epochs 5 --seed 42
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py \
--watermarking_method tramark \
--dataset cifar10 --num_clients 10 \
--alpha 0.04 \
--k 0.01 \
--local_epochs 5 \
--non_iid --gamma 0.5 --seed 42
```

Here,

| Argument        | Type       | Description   | Default Value |
|-----------------|------------|---------------|--------|
| `dataset`     | str | The main task dataset | cifar10|
| `num_clients`| str | The number of clients | 10|
| `alpha`         | str   | Warmup training ratio | 0.5|
| `k`    |   str     | The model partition ratio        | 0.01 |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `gamma`         | float | Data heterogeneous degree     | 0.5 |
| `seed` | int | Random seed | 42 |

For other arguments, you can check the `args.py` file, where the detailed explanation is presented.

## Citation

We provide the following BibTeX entry for citation for your convenience.

```
@inproceedings{
xu2026traceable,
title={Traceable Black-Box Watermarks For Federated Learning},
author={Jiahao Xu and Rui Hu and Olivera Kotevska and Zikai Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=xHRuyXnJXd}
}
