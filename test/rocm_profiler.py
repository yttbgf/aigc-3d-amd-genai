import torch
for i in range(10000):
    a = torch.ones((256,256),device=torch.device("cuda"))
    a = a*3
    b = a+100
    c = torch.mm(a,b)
    print(c)
