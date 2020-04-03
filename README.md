# TravelGAN
Implementation of " TraVeLGAN: Image-to-image Translation by Transformation Vector Learning " CVPR 2019 https://arxiv.org/abs/1902.09631

### Model 
![TravelGan](https://github.com/Medabid1/TravelGAN/master/imgs/model.png)

## Implementation
This Implementation is just One-Side Image Translation, Also i didn't use the same Generator (Unet) as described in the paper, instead i implemented a generator based on StarGan and adding to it a channel wise attention (introduced in Squeeze and Excitation paper)


## Usage 
1. Use the `train.py` file to train, just load ur data. First data loader passed is the source domain and the second is the target.
2. Use `config.ini` to change HyperParameters.


## Results