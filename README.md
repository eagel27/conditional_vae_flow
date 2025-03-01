# Conditional Variational Autoencoder with Normalizing Flow

This repository implements a Conditional Variational Autoencoder (CVAE) enhanced with Normalizing Flow to enable more precise sampling from the latent space.


## Overview

Variational Autoencoders (VAEs) are generative models that learn latent representations of data. By conditioning these models on additional information, Conditional VAEs (CVAEs) can generate data that adheres to specific attributes. Integrating Normalizing Flow into the VAE framework allows for more flexible and expressive latent distributions, improving the model's generative capabilities.

The code developed was used for this [conference paper](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_163.pdf).

## Project Setup & Execution  

### Installation  

To install the required dependencies, you can run:  
```bash
uv sync
```

And to run:  

```bash
uv run main.py
```
