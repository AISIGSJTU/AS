<div align="center">
  
# Improving Bayesian Neural Networks by Adversarial Sampling
[![Conference](http://img.shields.io/badge/AAAI-2022-4b44ce.svg)](https://aaai.org/Conferences/AAAI-22/) 

</div>
  
## Model training

    python3 main.py
    
Note that the default path to datasets is `~/data`. You can modify it in line [31](https://github.com/AISIGSJTU/AS/blob/main/main.py#L31), `main.py`.

## Environment
- python: 3.7.9

- torch: 1.7.1

- torchvision: 0.8.2

- numpy: 1.19.2

## Core codes
    
The Adversarial Sampling and calculation of the Adversarial loss is implemented in line [119](https://github.com/AISIGSJTU/AS/blob/main/utils.py#L119) - [152](https://github.com/AISIGSJTU/AS/blob/main/utils.py#L152), `utils.py`:

    # Initialize epsilon with random unit Gaussian variable
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                module.eps_weight.normal_()
                module.eps_weight.requires_grad = True
                if module.eps_bias is not None:
                    module.eps_bias.normal_()
                    module.eps_bias.requires_grad = True

    alpha = 0.02
    iters = 5


    for index in range(iters):
        for module in model.modules():
            if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                module.eps_weight.requires_grad = True
                if module.eps_bias is not None:
                    module.eps_bias.requires_grad = True

        routput, rkl = model(input_var, fix=True)
        model.zero_grad()
        rloss = - F.cross_entropy(routput, target) # Adversarial loss
        rloss.backward()
        # Updating
        with torch.no_grad():
            for module in model.modules():
                if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                    module.eps_weight -= alpha * module.eps_weight.grad.sign()
                    if module.eps_bias is not None:
                        module.eps_bias -= alpha * module.eps_bias.grad.sign()
    routput, rkl = model(input_var, fix=True)
    rloss = F.cross_entropy(routput, target) # Adversarial loss
    
## Cite

    @article{Zhang_Hua_Song_Wang_Xue_Ma_Guan_2022, 
        title={Improving Bayesian Neural Networks by Adversarial Sampling}, 
        volume={36}, 
        url={https://ojs.aaai.org/index.php/AAAI/article/view/21250}, 
        DOI={10.1609/aaai.v36i9.21250}, 
        number={9}, 
        journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
        author={Zhang, Jiaru and Hua, Yang and Song, Tao and Wang, Hao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing}, 
        year={2022}, 
        month={Jun.}, 
        pages={10110-10117} 
    }
