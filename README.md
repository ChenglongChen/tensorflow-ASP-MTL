# tensorflow-ASP-MTL

A Tensorflow implementation of Adversarial Shared-Private Model. It's used in
- Multi-Task Learning (ASP-MTL), see [1]
- Transfer Learning (ASP-TL), see [2]


To train the model:
```
cd data/
unzip mtl-dataset.zip
cd ../
python main.py
```

Use Tensorboard to monitor the training or testing process:
```
tensorboard --logdir=summary/train
```


# Reference
[1] Pengfei Liu, Xipeng Qiu and Xuanjing Huang, *Adversarial Multi-task Learning for Text Classification*

[2] Cen Chen, Yinfei Yang, Jun Zhou, Xiaolong Li and Forrest Sheng Bao, *Cross-Domain Review Helpfulness Prediction based on Convolutional Neural Networks with Auxiliary Domain Discriminators*

# Acknowledgments
This project gets inspirations from the following projects:
- [fudan_mtl_reviews](https://github.com/FrankWork/fudan_mtl_reviews)
- [tf-dann](https://github.com/pumpikano/tf-dann)
