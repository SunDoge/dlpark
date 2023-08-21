# Example for [dlpark](https://github.com/SunDoge/dlpark)

This is an example of how to transfer `image::RgbImage` to Python and how to transfer `torch.Tensor` to Rust.

## Usage

```shell
pip install maturin
maturin develop
pip install torch matplotlib
python main.py
```

| Input RGB Image     | Output BGR Image |
| ------------------- | ---------------- |
| ![candy](candy.jpg) | ![bgr](bgr.jpg)  |