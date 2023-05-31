import torch
import torch_scatter
import einops

PI = 2 * torch.acos(torch.tensor(0.))


class LambdaLayer(torch.nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)


class GNN(torch.nn.Module):
    def __init__(self, lam_dim=31, h_dim=32, layers=2, heads=8, pretrained=False):
        super().__init__()

        self.lam_dim = lam_dim
        self.h_dim = h_dim
        self.layers = layers
        self.heads = heads

        # input embedding
        self.emb = torch.nn.Sequential(
            torch.nn.Linear(lam_dim * 4, self.h_dim),
            LambdaLayer(lambda x: x.unsqueeze(-2).repeat((1, 1, self.heads, 1)))
        )

        # positional embedding
        self.pos_emb = torch.nn.Sequential(
            torch.nn.Linear(3, self.h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(self.h_dim),
            torch.nn.Linear(self.h_dim, self.h_dim),
            LambdaLayer(lambda x: x.unsqueeze(-2).repeat((1, self.heads, 1)))
        )

        # lstm parameters
        # cycle over layers
        for lth in range(1, self.layers + 1):
            # positive and negative directions
            for typ in ['p', 'm']:
                k_dim = 3 * self.h_dim

                # qkv linear projection weights
                setattr(self, 'Wqkv_%s_%d' % (typ, lth),
                        torch.nn.Parameter(torch.empty((self.heads, 3 * self.h_dim, k_dim)).uniform_(-1, 1)))

                # qkv activation and normalization
                setattr(self, 'qkv_%s_%d' % (typ, lth), torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    LambdaLayer(lambda x: einops.rearrange(x, 'b H (k h) -> b k H h', k=3)),
                    torch.nn.LayerNorm((3, self.heads, self.h_dim)),
                    LambdaLayer(lambda x: torch.unbind(x, -3))
                ))

                # v normalization
                setattr(self, 'vnorm_%s_%d' % (typ, lth), torch.nn.LayerNorm((self.heads, self.h_dim)))

                # linear projections for recurrent gates
                setattr(self, 'lin_%s_%d' % (typ, lth), torch.nn.Sequential(
                    torch.nn.Linear(self.h_dim, 4 * self.h_dim),
                    LambdaLayer(lambda x: torch.split(x, 4 * [self.h_dim], -1))
                ))

        self.hlin = torch.nn.Linear(2 * self.h_dim, 2 * self.h_dim)

        # mu and sigma normalization
        self.munorm = torch.nn.LayerNorm(self.h_dim)
        self.signorm = torch.nn.LayerNorm(self.h_dim)

        self.rnn = torch.nn.LSTM(self.h_dim * self.heads, self.h_dim * self.heads)

        # fully connected output layer
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim * self.heads, self.h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(self.h_dim),
            torch.nn.Linear(self.h_dim, self.lam_dim),
        )

        if pretrained:
            self.load_state_dict(torch.load('gnn_weights.pt',
                                            map_location=torch.device('cuda:0') if torch.cuda.is_available() else
                                            torch.device('cpu')))

    # input preprocessing
    @torch.no_grad()
    def preprocess(self, graph):

        # temporal features
        t = graph.time
        if len(t.shape) > 1:
            t = t[0]

        t = torch.stack([t, torch.sin(2 * PI * t), torch.cos(2 * PI * t)], 0)
        t = einops.rearrange(t, 'h (W w) -> W (h w)', w=self.lam_dim).unsqueeze(1).repeat((1, graph.x.shape[0], 1))

        # input signal
        x = einops.rearrange(graph.x, 'b (W w) -> W b w', w=self.lam_dim)

        # merged input (signal + temporal features)
        x = torch.cat([x, t], -1)

        return x

    # lstm states init
    def init_hidden(self, x):
        return torch.unbind(torch.zeros((4, x.shape[1], self.heads, self.h_dim)).to(x.device, torch.float32), 0)

    # lstm cell
    def cell(self, x, h, c, ei, ea, typ='p', layer=1):

        xhc = torch.cat([x, h, c], -1)

        xhc = torch.einsum('H h k, b H k -> b H h', getattr(self, 'Wqkv_%s_%d' % (typ, layer)), xhc)
        q, k, v0 = getattr(self, 'qkv_%s_%d' % (typ, layer))(xhc)

        v = v0[ei[0]]
        q = q[ei[0]]
        k = k[ei[1]]

        qk = torch.exp((q * k).mean(-1).unsqueeze(-1) + ea)
        v = torch_scatter.scatter_sum(qk * v, ei[1], dim=0, dim_size=x.shape[0])
        qk = torch_scatter.scatter_sum(qk, ei[1], dim=0, dim_size=x.shape[0])
        v = torch.where(qk > 0, v / qk, torch.tensor(0.).to(qk.device))

        v = getattr(self, 'vnorm_%s_%d' % (typ, layer))(torch.relu(v0 + v))

        o, g, f, i = getattr(self, 'lin_%s_%d' % (typ, layer))(v)

        # updated memory state
        nc = (torch.sigmoid(f) * c) + torch.sigmoid(i) * torch.tanh(g)

        # updated hidden state
        nh = torch.sigmoid(o) * torch.tanh(nc)

        # output state
        o = torch.relu(o + x)

        return nh, nc, o

    # graph reconstruction from latent tensor and positional embedding
    def upsample(self, x, pos_emb, batch):
        x_ = []
        for xb, b in zip(x.permute([1, 0, 2]), range(batch[-1] + 1)):
            x_.append(torch.einsum('p h, b h ->p b h', xb, pos_emb[batch == b]))
        return torch.cat(x_, 1)

    def forward(self, graph, particles=20):
        # input and edge indices
        x = self.preprocess(graph)

        # positional embedding
        pos_emb = self.pos_emb(graph.pos)

        # edge attribute
        ea = pos_emb[graph.edge_index[0]] - pos_emb[graph.edge_index[1]]

        # input embedding
        x = self.emb(x)

        # LSTM latent and memory cells
        hp, hm, cp, cm = self.init_hidden(x)

        # cycle over windows
        for Wp, Wm in zip(range(x.shape[0]), range(x.shape[0] - 1, -1, -1)):
            op = x[Wp]
            om = x[Wm]

            # cycle over layers
            for lth in range(1, self.layers + 1):
                hp, cp, op = self.cell(op, hp, cp, graph.edge_index, ea, typ='p', layer=lth)
                hm, cm, om = self.cell(om, hm, cm, graph.edge_index, ea, typ='m', layer=lth)

        h = einops.rearrange(torch.stack([hp, hm], -1), 'b H h k -> b H (h k)')
        h = self.hlin(h)

        # mu sigma tensor
        musig = einops.rearrange(h, 'b H (k2 h) -> k2 b H h', k2=2)
        musig = torch_scatter.scatter_sum(musig * pos_emb.unsqueeze(0), graph.batch, dim=1,
                                          dim_size=graph.batch[-1] + 1)

        # noise
        eps = torch.empty((particles, musig.shape[1], self.heads, self.h_dim)).normal_(0, 1).to(h.device)

        # latent representation
        o = self.munorm(musig[:1]) + torch.exp(self.signorm(musig[1:])) * eps
        o = o.reshape((particles, -1, self.h_dim * self.heads))

        # graph ensemble reconstruction
        o = self.upsample(o, pos_emb=pos_emb.reshape((-1, self.heads * self.h_dim)), batch=graph.batch)
        o = einops.rearrange(o, 'p b h -> (p b) h').unsqueeze(0).repeat((x.shape[0], 1, 1))
        o, (_, _) = self.rnn(o)
        o = self.fc(o)
        o = einops.rearrange(o, 'W (p b) w -> p b (W w)', p=particles)
        musig = einops.rearrange(musig, 'k b H h -> k b (H h)')

        return o, graph.x, musig

    @torch.no_grad()
    def score(self, graph, local=False):
        y, x, _ = self.forward(graph)

        s = (x - y.mean(0)).abs() / y.std(0)

        s = self.smoother(s).max(-1)[0]

        if local:
            return s

        s = torch_scatter.scatter_max(s, graph.batch, dim=0, dim_size=graph.batch[-1] + 1)[0]
        return s

    @torch.no_grad()
    def smoother(self, x):
        y = torch.nn.functional.pad(x.unsqueeze(1), (3, 3))
        y = torch.nn.functional.conv1d(y, torch.ones((1, 1, 7)).to(y.device) / 7.)
        return y[:, 0, :]


if __name__ == '__main__':
    model = GNN(pretrained=True)
