# the code is built upon https://github.com/evelinehong/slot-attention-pytorch

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def build_grid(resolution):
    """Function creates grid for position embedding
    Args:
        resolution (tuple): Image resolution

    Returns:
        numpy.array: Grid
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)

    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
            hidden_size: Size of input feature dimension.
            resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.resolution = resolution
        self.dense = nn.Linear(4, hidden_size, bias=True)
        self.grid = torch.Tensor(build_grid(resolution)).view((-1, 4)).unsqueeze(0).to("cuda:0")

    def forward(self, inputs):

        # spatial flatten [batch_size, channels, width, height] -> [batch_size, width * height, channels]
        inputs = inputs.view(*inputs.shape[:2], inputs.shape[-1]*inputs.shape[-2]).permute(0, 2, 1)
        embedding = self.dense(self.grid)

        return inputs + embedding


def unstack_and_split(input, batch_size):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = input.view((batch_size, -1, * input.shape[1:]))  # shape: [batch_size, num_slots, channels, width, height]

    mask = unstacked[:, :, 3, :, :].unsqueeze(2)
    channels = unstacked[:, :, :3, :, :]

    return channels, mask


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""

    slots = slots.view(-1, slots.shape[-1]).unsqueeze(-1).unsqueeze(-1)  # [batch_size*num_slots, slot_dim, 1, 1]
    return slots.tile((1, 1, *resolution))  # shape [batch_size*num_slots, slot_dim, height, width]


class SlotAttention(nn.Module):
    """This class implements slot attention for object discoverys

    Attributes:
        num_slots: Number of slots defined at test time (can be changed for inference)
    """

    def __init__(self, num_slots, num_iterations, input_dim, slot_dim, mlp_dim, device="cuda:0", implicit_diff=False, eps=1e-6):

        super(SlotAttention, self).__init__()

        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.device = device
        self.eps = eps
        self.implicit_diff = implicit_diff

        self.k_linear = nn.Linear(input_dim, slot_dim, bias=False)
        self.q_linear = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_linear = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.slots_layer_norm = nn.LayerNorm(slot_dim)
        self.mlp_layer_norm = nn.LayerNorm(input_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim)).to(self.device)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim)).to(self.device)
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, slot_dim))

    def step(self, slots, k, v, batch_size):
        slots_prev = slots
        slots = self.slots_layer_norm(slots)

        # Attention
        q = self.q_linear(slots)

        dots = torch.einsum('bid,bjd->bij', q, k)  # shape [batch_size, input_dim, slot_dim]
        attn = dots.softmax(dim=1) + self.eps  # shape [batch_size, input_dim, slot_dim]
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)  # shape [batch_size, input_dim, slot_dim]

        slots = self.gru(
            updates.contiguous().view(-1, self.slot_dim),
            slots_prev.contiguous().view(-1, self.slot_dim)
        )

        slots = slots.reshape(batch_size, -1, self.slot_dim)
        slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

    def forward(self, inputs, slot_init=None):

        # get input size
        batch_size, n, d_inputs = inputs.shape

        # apply layer norm to input
        inputs = self.input_layer_norm(inputs)

        # intial slot representation
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.num_slots, -1)

        # slots shape [batch_size, num_slots, num_inputs]
        slots = mu + sigma * torch.randn(mu.shape, device=self.device)

        if slot_init is not None:
            slots = slot_init

        # apply linear transformations
        k = self.k_linear(inputs) * self.slot_dim ** -.5  # shape [batch_size, height*width, input_dim]
        v = self.v_linear(inputs)  # shape [batch_size, height*width, input_dim]

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots = self.step(slots, k, v, batch_size)

        # object representation as fixed point paper
        if self.implicit_diff:
            slots = self.step(slots.detach(), k, v, batch_size)

        return slots


class SlotAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, device, resolution=(128, 128), num_slots=15, slot_dim=64, num_iterations=5, vae=False, implicit_diff=False, encoder=None):
        """_summary_

        Args:
            device: Cpu or Cuda
            resolution (tuple, optional): Resolution of the input images. Defaults to (64, 64).
            num_slots (int, optional): Number of individual slots in Slot Attention. Defaults to 10.
            slot_dim (int, optional): Size of one Slot. Defaults to 64.
            num_iterations (int, optional): Number of iterations f. Defaults to 5.
            implicit_diff (bool): Use implicit differentiation
        """

        super(SlotAutoEncoder, self).__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.vae = vae

        self.enc_channels = 64

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.enc_channels, kernel_size=5, padding='same'),
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=slot_dim, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=3, stride=1))

        self.decoder_init = (8, 8)

        self.encoder_pos = SoftPositionEmbed(hidden_size=self.enc_channels, resolution=resolution)  # resolution = output shape of encoder
        self.decoder_pos = SoftPositionEmbed(hidden_size=self.slot_dim, resolution=self.decoder_init)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.enc_channels, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2*self.enc_channels if self.vae else self.enc_channels)
        )

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            num_iterations=self.num_iterations,
            input_dim=self.enc_channels,
            slot_dim=self.slot_dim,
            mlp_dim=128,
            device=device,
            implicit_diff=implicit_diff)

        self.input_layer_norm = nn.LayerNorm(self.enc_channels)

    def forward(self, input, slot_init=None):

        batch_size, _, _, _ = input.shape

        # CNN backbone
        z = self.encoder_cnn(input)  # shape: [batch_size, channels, width, height]

        # add positional encoding
        z = self.encoder_pos(z)  # shape: [batch_size, width * height, input_dim]

        # MLP head
        features = self.mlp(self.input_layer_norm(z))  # shape: [batch_size, width * height, input_dim]

        if self.vae:
            mu = features[..., :self.enc_channels]
            logvar = features[..., self.enc_channels:]
            features = reparametrize(mu, logvar)

        # Slot Attention
        if slot_init is not None:
            slots = self.slot_attention(features, slot_init)
        else:
            slots = self.slot_attention(features)  # shape: [batch_size, num_slots, slot_dim]

        z = spatial_broadcast(slots, self.decoder_init)  # shape: [batch_size*num_slots, slot_dim, height_init, width_init]
        z = self.decoder_pos(z)

        z = z.permute(0, 2, 1)  # shape: [batch_size * num_slots, slot_dim, width_init * height_init]
        z = z.view(*z.shape[:2], *self.decoder_init)  # shape: [batch_size * num_slots, slot_dim, width_init, height_init]

        # CNN Decoder
        z = self.decoder_cnn(z)

        # Reconstructions, Alpha Mask
        recons, masks = unstack_and_split(z, batch_size=batch_size)

        recon_combined = torch.sum(masks.softmax(dim=1) * recons, dim=1)

        if self.vae:
            return recon_combined, recons, masks, slots, mu, logvar
        else:
            return recon_combined, recons, masks, slots, features

    def encode(self, input):
        return self.encoder_cnn(input)

    def decode(self, input):
        return self.decoder_cnn(input)


class SlotBase(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, device, resolution=(128, 128), num_class=10, num_slots=15, slot_dim=64, num_iterations=5, vae=False, implicit_diff=False, encoder=None):
        """_summary_

        Args:
            device: Cpu or Cuda
            resolution (tuple, optional): Resolution of the input images. Defaults to (64, 64).
            num_slots (int, optional): Number of individual slots in Slot Attention. Defaults to 10.
            slot_dim (int, optional): Size of one Slot. Defaults to 64.
            num_iterations (int, optional): Number of iterations f. Defaults to 5.
            implicit_diff (bool): Use implicit differentiation
        """

        super(SlotBase, self).__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.vae = vae

        self.enc_channels = 64

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.enc_channels, kernel_size=5, padding='same'),
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=slot_dim, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=3, stride=1))

        self.decoder_init = (8, 8)

        self.encoder_pos = SoftPositionEmbed(hidden_size=self.enc_channels, resolution=resolution)  # resolution = output shape of encoder
        self.decoder_pos = SoftPositionEmbed(hidden_size=self.slot_dim, resolution=self.decoder_init)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.enc_channels, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2*self.enc_channels if self.vae else self.enc_channels)
        )

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            num_iterations=self.num_iterations,
            input_dim=self.enc_channels,
            slot_dim=self.slot_dim,
            mlp_dim=128,
            device=device,
            implicit_diff=implicit_diff)

        self.input_layer_norm = nn.LayerNorm(self.enc_channels)

        # freeze all layers
        for name, param in self.encoder_cnn.named_parameters():
            param.requires_grad = False

        for name, param in self.encoder_pos.named_parameters():
            param.requires_grad = False

        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        for name, param in self.slot_attention.named_parameters():
            param.requires_grad = False

        for name, param in self.decoder_cnn.named_parameters():
            param.requires_grad = False

        for name, param in self.decoder_pos.named_parameters():
            param.requires_grad = False

        for name, param in self.input_layer_norm.named_parameters():
            param.requires_grad = False

        # downstream task
        self.proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.pairwise = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(64, num_class)

    def encode_single(self, input, slot_init=None):

        # CNN backbone
        z = self.encoder_cnn(input)  # shape: [batch_size, channels, width, height]

        # add positional encoding
        z = self.encoder_pos(z)  # shape: [batch_size, width * height, input_dim]

        # MLP head
        features = self.mlp(self.input_layer_norm(z))  # shape: [batch_size, width * height, input_dim]

        # Slot Attention
        if slot_init is not None:
            slots = self.slot_attention(features, slot_init)
        else:
            slots = self.slot_attention(features)  # shape: [batch_size, num_slots, slot_dim]

        return slots, features

    def encode_pair(self, x1, x2):
        slot1, feat1 = self.encode_single(x1)
        slot2, feat2 = self.encode_single(x2, slot1)
        slot_init = (slot1 + slot2) / 2.0
        slot1, feat1 = self.encode_single(x1, slot_init)
        slot2, feat2 = self.encode_single(x2, slot_init)
        return slot1, slot2


class SlotMatchMax(SlotBase):
    def __init__(self, device, resolution=(128, 128), num_class=10):
        super().__init__(device, resolution, num_class)
        print('@ SlotMatchMax')

    def forward(self, x1, x2, slot_init=None):
        # encode
        slot1, slot2 = self.encode_pair(x1, x2)

        # action
        # p1 = self.proj(slot1)
        # p2 = self.proj(slot2)
        # z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).max(dim=1)[0]

        p1 = self.proj(slot1.view(-1, 64))
        p2 = self.proj(slot2.view(-1, 64))
        z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).view(-1, self.num_slots, 64).max(dim=1)[0]

        # output
        logit = self.classifier(z)

        return logit, z, p1, p2


class SlotMatchMean(SlotBase):
    def __init__(self, device, resolution=(128, 128), num_class=10):
        super().__init__(device, resolution, num_class)
        print('@ SlotMatchMean')

    def forward(self, x1, x2, slot_init=None):
        # encode
        slot1, slot2 = self.encode_pair(x1, x2)

        # action
        # p1 = self.proj(slot1)
        # p2 = self.proj(slot2)
        # z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).mean(dim=1)

        p1 = self.proj(slot1.view(-1, 64))
        p2 = self.proj(slot2.view(-1, 64))
        z = self.pairwise(torch.cat([p1, p2-p1], axis=-1)).view(-1, self.num_slots, 64).mean(dim=1)

        # output
        logit = self.classifier(z)

        return logit, z, p1, p2


class SlotAverage(SlotBase):
    def __init__(self, device, resolution=(128, 128), num_class=10):
        super().__init__(device, resolution, num_class)
        print('@ SlotAverage')

    def forward(self, x1, x2, slot_init=None):
        # encode
        slot1, slot2 = self.encode_pair(x1, x2)
        feat1 = slot1.mean(dim=1)
        feat2 = slot2.mean(dim=1)

        # action
        p1 = self.proj(feat1)
        p2 = self.proj(feat2)
        z = self.pairwise(torch.cat([p1, p2-p1], axis=-1))

        # output
        logit = self.classifier(z)

        return logit, z, p1, p2
