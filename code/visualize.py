import pickle
import pylab
import matplotlib.pyplot as plt
f=open(r"../data/test.p", "rb")
f1=open(r"fgsm/fgsm.pkl", "rb")
f2=open(r"jsma/jsma.pkl", "rb")
f3=open(r"l-bfgs/bfgs.pkl", "rb")
img=(pickle.load(f))["features"]
img1=(pickle.load(f1))["features"]
img2=(pickle.load(f2))["features"]
img3=(pickle.load(f3))["features"]
import numpy as np
import pylab
import matplotlib.cm as cm
import time
dic={0:"original",1:"fgsm",2:"jsma",3:"l-bfgs"}
num=0
for item in zip(img,img1,img2,img3):
    fx = pylab.figure()
    for n, fname in enumerate(item):
        arr=np.asarray(fname)
        ax=fx.add_subplot(2, 2, n+1)  # this line outputs images on top of each other
        ax.set_title(dic[n])
        pylab.imshow(arr,cmap=cm.Greys_r)
        pylab.xticks([])
        pylab.yticks([])
    pylab.savefig(str(num) + ".jpg")
    pylab.show()

    time.sleep(10)
    num=num+1
    if num==10:
        break
