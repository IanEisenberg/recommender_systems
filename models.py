import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users:int, n_items:int, n_factors:int=50, bias=True,
                y_range=None):
        super().__init__()
        self.y_range = y_range
        self.user_embedding = torch.nn.Embedding(n_users+1, n_factors)
        self.item_embedding = torch.nn.Embedding(n_items+1, n_factors)
        self.bias = bias
        if bias:
            self.user_biases = torch.nn.Embedding(n_users+1, 1)
            self.item_biases = torch.nn.Embedding(n_items+1, 1)
        self._init()
        
    def forward(self, users, items):
        out =  (self.user_embedding(users) * self.item_embedding(items)).sum(1)
        if self.bias:
            out += (self.user_biases(users) + self.item_biases(items)).squeeze()
        if self.y_range is None: return out
        return torch.sigmoid(out) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
    
    def _init(self): 
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)
        if self.bias:
            self.user_biases.weight.data.uniform_(-0.05, 0.05)
            self.item_biases.weight.data.uniform_(-0.05, 0.05)
            
class NCF(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=50,
                linear_layers = [10],
                embedding_dropout=.05,
                linear_dropout=.25,
                y_range=None):
        super().__init__()
        self.y_range = y_range
        self.user_embedding = torch.nn.Embedding(n_users+1, n_factors)
        self.item_embedding = torch.nn.Embedding(n_items+1, n_factors)
        self.feature_drop = torch.nn.Dropout(embedding_dropout)
        def gen_layers(input_size, layer_sizes, dropout):
            layers = []
            for size in layer_sizes:
                layers.append(torch.nn.Linear(input_size, size))
                input_size = size
                layers.append(torch.nn.ReLU())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
            return layers

        self.linear = None
        if len(linear_layers) > 0:
            self.linear = torch.nn.Sequential(*gen_layers(n_factors*2,
                                                        linear_layers,
                                                        linear_dropout))
            self.fc  = torch.nn.Linear(linear_layers[-1],1)
        else:
            self.fc = torch.nn.Linear(n_factors*2, 1)

    def forward(self, users, items):
        cat = torch.cat([self.user_embedding(users), self.item_embedding(items)], 1)
        out = self.feature_drop(cat)
        if self.linear is not None:
            out = self.linear(out)
        out = self.fc(out)
        if self.y_range is None: return out.squeeze()
        out = torch.sigmoid(out) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
        return out.squeeze()

    def _init(self):
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)
        self.linear.apply(init)
        init(self.fc)


