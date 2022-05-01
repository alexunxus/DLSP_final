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

Using different loss function   

`TODO: should provide a graph`  

Using different batch size(i.e. number of negative example)  

`TODO: should provide a graph`  
