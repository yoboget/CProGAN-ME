# CProGAN-ME
Condional Progressive GAN with Multiple Encoders

Warning: 

Most of the code provided here comes from the work of Tero Karras (NVIDIA), Timo Aila (NVIDIA), Samuli Laine (NVIDIA), Jaakko Lehtinen (NVIDIA and Aalto University). The original paper 'Progressive Growing of GANs for Improved Quality, Stability, and Variation' is available here: https://arxiv.org/abs/1710.10196
Their Official TensorFlow implementation of the ICLR 2018 paper is here: https://github.com/tkarras/progressive_growing_of_gans. 
Please refer to the original paper for the implementation. 

I am accoutable only for the conditional part of it.

Networks.py - Lines: 182-224, Modified Generator receiving the input from the encoders

Networks.py - Lines: 225-265, Encoders

loss.py     - Lines: 25-65, Modification of the input of the loss function to make it conditional

Train.py    - Lines: 133-330, Adapted Diognostic tools

![Conditional_tiles](conditional_tiles.png?raw=true "Conditional tiles")


Abstract of the research
(Full paper soon available)

In this paper, we propose a new approach for image in-painting and image extension: the Conditional Progressive GAN with Multiple Encoders C-ProGAN-ME. Image in-painting and image extension can typically be represented in terms of conditional probabilities. Given the known part of the image, what are likely values for the missing part? Unlike most previous works using a combination of reconstruction, contextual, or/and adversarial losses, our method is strictly conditional and only (Condiational)GAN-based so that it preserves the key-feature of CGAN: the ability to generate samples approximating the full predictive distribution for a new observation.

The major issue resulting from keeping a strict GAN architecture lies in the fact that the conditional part is too large to be directly injected as input into the Generator. It should be encoded, creating a bottleneck limiting the quantity of information transmitted to the Generator. To tackle this issue, we introduced secondary encoders re-injecting information at each stage of the Progressive Generator. Even if we still have to assess our method on standard datasets, we show that it performs realistic image completion. Also, we can use the technique iteratively to produce larger images. 

![Image_iterative_CProGAN-ME](image3.png?raw=true "Iterative use of CProGAN-ME")
