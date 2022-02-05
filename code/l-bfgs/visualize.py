import pickle
import pylab
import matplotlib.pyplot as plt
f=open(r"bfgs.pkl","rb")
img=pickle.load(f)
print(len(img["features"]))
for item in img["features"]:
    pylab.imshow(item)
    pylab.gray()
    pylab.show()
    # break
f.close()