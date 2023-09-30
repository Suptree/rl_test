import torch
from tensordict import TensorDict


batch_size = 5
tensordict = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 3),
        "key 2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)
print(tensordict)


print(tensordict[2])


print(tensordict["key 1"] is tensordict.get("key 1"))