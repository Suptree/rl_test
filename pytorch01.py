import torch
import numpy as np


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

np_array = np.array(data)
print(np_array)
x_np = torch.from_numpy(np_array)
print(x_np)


x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# GPUが使用可能であれば、GPU上にテンソルを移動させる
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


# 2つのテンソル行列のかけ算です。 y1, y2, y3 は同じ結果になります。
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# こちらは、要素ごとの積を求めます。 z1, z2, z3 は同じ値になります。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)