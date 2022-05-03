# Practical Deep Learning System Performance Final Project

Team member:  
1. kh3120 Kuanyao Huang
2. sm5116 Sujith Reddy Mammidi

## Project description

Machine learning robustness is now a popular topic in computer vision. In this project, we combine two 
techniques: **self-supervised learning**(SSL) and **contrastive loss** to improve model's performance on corrupted 
datasets at **inference time**.

We reimplemented the paper *Adversarial Attacks are Reversible with Natural Supervision* by Chengzhi Mao et al. 
The basic idea is that adversarial attack can greatly decrease the mutual information between the input data and
the label. In addition, an increase in contrastive loss between augmentations of the same image is observed due 
to the adversarial attack. The author proposed that if we can "correct" the image through self-supervised learning,
then we can prove the mutual information between the image and label improves and hence leads to a better performance.

The advantage of this method is that we don't have to retrain the clean classifier, we correct the image at test time
instead. The main drawback of this method is this method spends more time to perform inference. 

Our project can be divided into several part:
1. Train the clean classifier
2. Prepare the adversarial dataset
3. Perform self-supervised learning to train the self-supervised learning head
4. Perform inference on the adversarial dataset


----------------------------
# Implementation Details

## Part 1: Train the clean classifier
Train the clean classifier using wide-resnet34  
The weight can be downloaded by

```
mkdir weight
cd weight/
wget https://cv.cs.columbia.edu/mcz/ICCVRevAttack/cifar10_rst_adv.pt.ckpt
```

## Part 2: Create adversarial dataset
We adopt projected gradient descent to generate the adversarial attacks on test images and 
store them in a temporary folder `./data/pgd/`
```
python3 src/attack.py --attack_iters 5 --norm l_1
python3 src/attack.py --attack_iters 5 --norm l_2
python3 src/attack.py --attack_iters 5 --norm l_inf
```
or
```
python3 src/attack.py --attack_iters 5 --norm all
```
You can specify number of iterations for pgd attack.

### Creating adversarial test datasets using pgd and Visualization

`TODO: provide some viusalization example`

## Part 3: Train Self-Supervised Head
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

## Part 5: Inference with SSL
```
python3 inference.py --task SSL --norm clean
python3 inference.py --task SSL --norm l_1
python3 inference.py --task SSL --norm l_2
python3 inference.py --task SSL --norm l_inf
```



Using different loss function   

`TODO: should provide a graph`  

Using different batch size(i.e. number of negative example)  

`TODO: should provide a graph`  
