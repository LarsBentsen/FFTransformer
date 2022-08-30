# FFTransformer

We propose the FFTransformer, a new alteration of the Transformer architecture. The architecture is based on wavelet decomposition and an adapted Transformer architecture consisting of two streams. One stream analyses periodic components ni the frequency domain with an adapted attentiom mechanism based on fast Fourier Transform (FFT), and another stream similar to the vanilla Transformer, which leanrs trend components. So far the model have only been tested for spatio-temporal multi-step wind speed forecasting [[paper](https://arxiv.org/abs/2106.13008)]. This repository contains code for our proposed model, the [Transformer](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), [LogSparse Transformer](https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html), [Informer](https://arxiv.org/abs/2012.07436), [Autoformer](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html), LSTM, MLP and a persistence model. All architectures are also implemented in a spatio-temporal setting, where the respective models are used as update functions in GNNs. Scripts for running the models follow the same style as in the [Autoformer repo](https://github.com/thuml/Autoformer). 

An overview of the methodology used in the [paper](https://arxiv.org/abs/2106.13008), can be seen in Fig. 1, along with the FFTransformer model, shown for an encoder-decoder setting, in Fig. 2.

<p align="center">
<img src=".\pic\GraphicalAbstract.jpg" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Graphical Abstract.
</p>

<p align="center">
<img src=".\pic\FFTransformer.jpg" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> FFTransformer Model.
</p>


## Dataset

The dataset contained off-shore metereological measurements from the Norwegian continental shelf. The data was made available by the Norwegian Meterological Institue and can be downloaded using the [Frost API](https://frost.met.no/index.html).

## Citation

If you found this repo useful, please cite our paper. 

```
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Contact

If you have any question or want to use the code, please contact l.o.bentsen@its.uio.no .

## Acknowledgement

We appreciate the following github repos for their publicly available code and methdos:


https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/mlpotter/Transformer_Time_Series

https://github.com/fbcotter/pytorch_wavelets

https://github.com/TQCAI/graph_nets_pytorch 

