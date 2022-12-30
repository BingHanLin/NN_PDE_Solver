# https: // machinelearningmastery.com/calculating-derivatives-in-pytorch/

import matplotlib.pyplot as plt
import torch

x = torch.tensor(3.0, requires_grad=True)
print("creating a tensor x: ", x)

y = 3 * x ** 2
print("Result of the equation is: ", y)
y.backward()
print("Dervative of the equation at x = 3 is: ", x.grad)

print('data attribute of the tensor:', x.data)
print('grad attribute of the tensor::', x.grad)
print('grad_fn attribute of the tensor::', x.grad_fn)
print("is_leaf attribute of the tensor::", x.is_leaf)
print("requires_grad attribute of the tensor::", x.requires_grad)

print('=================================')

print('data attribute of the tensor:', y.data)
# print('grad attribute of the tensor:', y.grad)
print('grad_fn attribute of the tensor:', y.grad_fn)
print("is_leaf attribute of the tensor:", y.is_leaf)
print("requires_grad attribute of the tensor:", y.requires_grad)
