import torch
import torch.nn as nn

class Spatial(nn.Module):
    def __init__(self, feature, input_dim):
        super(Spatial, self).__init__()
        dmodel = input_dim * feature
        self.scale = (dmodel) ** -0.5

        self.q = nn.Linear(dmodel, dmodel)
        self.k = nn.Linear(dmodel, dmodel)
        self.v = nn.Linear(dmodel, dmodel)
        self.out = nn.Linear(dmodel, dmodel)

        self.linear1 = nn.Linear(dmodel,dmodel*4)
        self.linear2 = nn.Linear(dmodel*4,dmodel)
        self.act = nn.ReLU()

    def forward(self, x, graph): # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.transpose(1,2).reshape(B, N, T*D) # (B, N, T, D) -->  (B, N, T*D)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # (B, N, T*D) * (B, T*D, N) --> (B, N, N)
        attention = (torch.matmul(q, k.transpose(-2, -1))) * self.scale

        # 稀疏化操作 (B, N, N)
        k = 16 # int( N / 16)
        sparse_attention = self.top_k_(attention, k)
        sparse_attention = sparse_attention.softmax(dim=-1)


        # (B, N, N) * (B, N, T*D) --> (B, N, T*D)
        sparse_out = torch.matmul(sparse_attention, v)
        sparse_out = self.out(sparse_out) # (B, N, T*D)

        if graph is None:
            graph_out = 0
        else:
            # graph_mask操作, (N, N) * (B, N, N) --> (B, N, N)
            graph_mask = torch.matmul(graph, attention.transpose(-2,-1))
            graph_mask.data.masked_fill_(torch.eq(graph_mask, 0), float('-inf'))
            graph_mask = graph_mask.softmax(dim=-1)

            #  (B, N, N) * (B, N, T*D') -->  # (B, N, T*D')
            graph_out = self.act(torch.matmul(graph_mask, self.linear1(x))) # (B, N, T*D')
            graph_out = self.act(torch.matmul(graph_mask, self.linear2(graph_out)))  # (B, N, T*D)

        out = sparse_out + graph_out # (B, N, T*D)

        # (B, N, T*D) -- > (B, N, T, D) --> (B, T, N, D)
        out = out.reshape(B, N, T, D).transpose(1,2)

        return out # (B, T, N, D)

    def top_k_(self,x, k): # (B, N, N)
        b, n, n = x.shape
        values, indices = torch.topk(x, k, dim=-1)
        x_min = torch.min(values, dim=-1).values

        x_min = x_min.unsqueeze(-1).repeat(1, 1, n)
        ge = torch.ge(x, x_min)
        zero = torch.zeros_like(x)
        attention = torch.where(ge, x, zero)
        attention.data.masked_fill_(torch.eq(attention, 0), float('-inf'))
        return attention

class FC(nn.Module):
    def __init__(self, input_dim):
        super(FC, self).__init__()

        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x): # (B, T, N, D)
        x = x.transpose(1,3)
        x = self.linear(x)
        x = x.transpose(1, 3)

        return x # (B, T, N, D)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = x[:, :, :-self.chomp_size]

        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, groups, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, groups = 1)
        self.chomp1 = Chomp(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net_u = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

    def forward(self, x):
        x = self.net_u(x)
        return x

class BTCN(nn.Module):
    def __init__(self, nodes, groups, dropout):
        super(BTCN, self).__init__()

        kernel_size = 2
        in_channels = nodes
        out_channels = nodes
        stride = 1
        layer1 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=1, padding= 1, dropout=dropout) # padding=(kernel_size - 1) * 1
        layer2 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=2, padding= 2, dropout=dropout)
        layer3 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)
        layer4 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)

        self.network_p = nn.Sequential(layer1, layer2, layer3, layer4)
        self.network_b = nn.Sequential(layer1, layer2, layer3, layer4)

    def forward(self, x): # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.reshape(B, T, N*D)

        x = x.transpose(1,2) # (B, N*D, T)
        x_p = self.network_p(x)

        x_re = torch.flip(x,dims=[2]) #  (B, N*D, T)
        x_b = self.network_b(x_re)
        x_b = torch.flip(x_b,dims=[2])
        x = x_p + x_b
        x = x.transpose(1,2) # (B, T, N*D)
        x = x.reshape(B, T, N, D)
        return x

class ST_Block(nn.Module):
    def __init__(self, adj_data,  f_nodes, s_nodes, feature, input_dim, hidden_dim, output_dim, dropout):
        super(ST_Block, self).__init__()

        self.graph = adj_data
        self.f_nodes = f_nodes
        self.s_nodes = s_nodes
        self.nodes = f_nodes + s_nodes

        self.spatial_f = Spatial(feature, input_dim)
        self.spatial_s = Spatial(feature, input_dim)

        self.spatial = Spatial(feature, input_dim)

        self.btcn_f = BTCN(f_nodes * feature, groups=self.f_nodes * feature, dropout=dropout)
        self.btcn_s = BTCN(s_nodes * feature, groups=self.s_nodes * feature, dropout=dropout)

        self.btcn = BTCN(self.nodes * feature, groups=self.nodes * feature, dropout=dropout)


    def forward(self,x): # (B, T, N, D)
        res = x

        s_cr = self.spatial(x, self.graph)

        xs = s_cr + res
        t1 = self.btcn(xs)

        t = t1 + xs

        t_f = t[:, :, :self.f_nodes, :]
        t_s = t[:, :, self.f_nodes:, :]
        t_f = self.btcn_f(t_f)
        t_s = self.btcn_s(t_s)
        t2 = torch.cat([t_f, t_s], dim=2)

        xt = t2

        return xt

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x): # (B, T', N, D)
        pass
        x = x.transpose(1,3)
        x = self.linear(x)
        x = x.transpose(1, 3)

        return x # (B, T, D, N)

class Network(nn.Module):
    def __init__(self, adj_data, f_nodes, s_nodes, blocks, feature, input_dim, hidden_dim, output_dim, dropout):
        super(Network, self).__init__()
        self.f_nodes = f_nodes
        self.s_nodes = s_nodes
        nodes = f_nodes + s_nodes

        self.blocks = blocks
        st_blocks = []
        for i in range(self.blocks):
            st_blocks.append(ST_Block(adj_data, f_nodes, s_nodes, feature, input_dim, hidden_dim, output_dim, dropout))
        self.st = nn.ModuleList(st_blocks)

        self.mlp = MLP(input_dim * (blocks + 1), hidden_dim, output_dim, dropout=dropout)

    def forward(self, x): # (B, T, N, D)
        res = x
        ST_out = []
        st_out = x
        for st in self.st:
            st_out = st(st_out)
            ST_out.append(st_out)
            st_out = st_out + res

        ST_out.append(res)

        out = torch.cat(ST_out,dim=1) # (B, T', N, D)
        x = self.mlp(out)
        return x

