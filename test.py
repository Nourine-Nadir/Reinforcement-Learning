import numpy as np
# k = np.arange(6400).reshape((1,80,80))
# print(k)
# print(np.moveaxis(k,2,0))
# print(k.repeat(4,axis=0).shape)

# k = np.arange(20).reshape((4,5))
# print(k)
# print(k[::3,::2])

# print(np.zeros_like(4,80,80))

# t = np.arange(11)
# print(t)
#
# k= list(filter(lambda d: d>3,t) )
#
# p = [d for d in t if d>3]
# print(k)
# print(p)
#
# H = [1,2,3,3,4,5,1,2,3,5,5,1,3,3,5,6,8,9,9,1
#      ]
#
# print(any(H[i] == 3 and H[i+1]==3 for i  in range(len(H)-1)))
#
# rewards = np.arange(10)
# print(rewards[::-1])

# list_of_arrays = [
#     np.array([1, 2]),
#     np.array([3, 4]),
#     np.array([5, 6])
# ]
# print(list_of_arrays)
# # np.stack will combine them into a single array
# stacked = np.stack(list_of_arrays)
# print(stacked)
# # Output:
# # array([[1, 2],
# #        [3, 4],
# #        [5, 6]])
#
# print(stacked.shape)  # (3, 2)

t = np.random.uniform(low=-1, high=1, size=4)
print(t)

p, o = 0,1

print(p,o)