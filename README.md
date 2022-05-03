# Practical Deep Learning System Performance Final Project

Team member:  
1. kh3120 Kuanyao Huang
2. sm5116 Sujith Reddy Mammidi

## Part 1: Train the clean classifier
Train the clean classifier using wide-resnet34  
The weight can be downloaded by

```
mkdir weight
cd weight/
wget https://cv.cs.columbia.edu/mcz/ICCVRevAttack/cifar10_rst_adv.pt.ckpt
```

## Part 2: Create adversarial dataset
Creating adversarial test datasets using pgd and Visualization

`TODO: provide some viusalization example`

## Part 3: Train Clean Classifier and Self-Supervised Model
Command:
```
python3 train.py --task SSL
```

## Part 4: Inference without SSL:
```
python3 inference.py --norm clean
python3 inference.py --norm l_1
python3 inference.py --norm l_2
python3 inference.py --norm l_inf
```

Result: 
| Perturbation | Accuracy (%) | Test Loss |
|--------------|--------------|-----------|
| Clean        | 87.52        | 0.5812    |
| L1           | 70.47        | 0.8985    |
| L2           | 10.58        | 5.0381    |
| Linf         | 6.66         | 4.3066    |

Using different loss function   

`TODO: should provide a graph`  

Using different batch size(i.e. number of negative example)  

`TODO: should provide a graph`  
