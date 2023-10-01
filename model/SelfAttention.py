import torch.nn as nn
import torch
import math

class Self_Attention_Muti_Head(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v,nums_head):
        super(Self_Attention_Muti_Head,self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        
        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    def forward(self,x):
        # print('=====attention in: ',x.shape) #b n d
        Q = self.q(x).reshape(x.shape[0],x.shape[1],-1,self.dim_k // self.nums_head).permute(2,0,1,3) # b n h k/h # h b n k/h
        K = self.k(x).reshape(x.shape[0],x.shape[1],-1,self.dim_k // self.nums_head).permute(2,0,1,3) # b n h k/h # h b n k/h
        V = self.v(x).reshape(x.shape[0],x.shape[1],-1,self.dim_v // self.nums_head).permute(2,0,1,3) # b n h v/h # h b n v/h
        atten = nn.Softmax(dim=-1)(torch.matmul(Q,K.permute(0,1,3,2))) # Q * K.T() # h b n n
        output = torch.matmul(atten,V).permute(1,2,0,3).reshape(x.shape[0],x.shape[1],-1) # Q * K.T() * V # h b n v/h # b n v
        # print('=====attention finish: ',output.shape)
        # x = output.tranpose(0,1)
        # print('=====attention tranpose finish: ',x.shape)
        return output #, atten