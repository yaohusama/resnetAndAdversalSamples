1、运行环境是：windows11
2、软件版本是：tensorflow=1.14;python=3.7;
3、要生成fgsm方法生成的对抗样本则是，python fgsm.py生成fgsm.pkl作为gtsrb测试集的对抗样本，训练好的模型参数在ckpt文件夹中；
要生成sjma方法生成的对抗样本运行命令：python sjma.py生成sjma.pkl作为gtsrb测试集的对抗样本，训练好的模型在ckpt_jsma文件夹中；
4、python visualize.py可以对fgsm.pkl等对应的对抗样本生成图片，可以验证是否生成了合理的对抗样本。
3、运行l-bfgs步骤：python howToGenerateAdExam.py产生对抗样本文件为bfgs.pkl;
python visualize.py 可以可视化展示bfgs.pkl中的图片。
读取同目录下的data文件夹下的gtsrb的test.p数据集，生成对抗样本后，对对抗样本进行可视化。
4、保存的模型参数在training_2文件夹下，ckpt断点在这个文件夹内。