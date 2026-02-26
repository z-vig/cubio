import numpy as np

mylist = np.array(["thing1", "thing2", "thing3"])

mytuple = tuple(mylist[[1, 2, 0]])

print(mytuple)
