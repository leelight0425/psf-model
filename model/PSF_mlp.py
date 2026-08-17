# --coding:utf-8--
import torch.nn as nn
import torch
from torch.fft import fftshift, fft2
import torchvision


class seidel2wavefront(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(seidel2wavefront, self).__init__()
        self.device = device

    def forward(self, seidel, IS, color, BS):
        # BS = s
        M = IS.wf_res[color]
        sel_basis = IS.seidel_basis[color]
        num_seidel = sel_basis[color].shape[-1]
        seidel1 = seidel.unsqueeze(1).unsqueeze(1)
        seidel2 = seidel1.repeat(1, M, M, 1).to(self.device)
        WF = torch.zeros(BS, M, M).to(self.device)
        A = sel_basis.unsqueeze(0).repeat(BS, 1, 1, 1).to(self.device)
        x = torch.linspace(-1, 1, M).to(self.device)
        Y, X = torch.meshgrid(x, x, indexing='ij')
        rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2)).to(self.device)
        rho = rho.repeat(BS, 1, 1)
        # print(A.shape)
        # print(seidel2.shape)
        for i in range(num_seidel):
            WF = WF + A[..., i] * seidel2[..., i]
        WF = torch.where(rho >= 1, 0, WF)
        return WF


class wavefront2psf(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(wavefront2psf, self).__init__()
        self.device = device

    def forward(self, WF):
        M = WF.size(1)
        W = nn.ZeroPad2d(2 * M)(WF)
        W = W
        phase = torch.exp(-1j * 2 * torch.pi * W)
        phase = torch.where(phase == 1, 0, phase)
        # clq
        # print(phase.shape)
        phase = fft2(phase)
        phase = fftshift(phase)
        AP = abs(phase) ** 2
        # AP = abs(fftshift(fft2(phase))) ** 2
        CenterCrop = torchvision.transforms.CenterCrop(M)
        AP = CenterCrop(AP)
        AP = AP / torch.max(AP)
        return AP.to(self.device)


class PSF_mlp(nn.Module):
    """
    Definition of a PSF generation mlp (Multilayer Perceptron).
    - input:
    - output: coefficients of wavefront basis, wavefront, PSF (Point Spread Function)
    """
    def __init__(self, device=torch.device('cuda')):
        super(PSF_mlp, self).__init__()
        self.device = device
        self.fc1_0 = nn.Sequential(nn.Linear(6, 10), nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(10, 10),
                                 nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(10, 6)).to(device)
        self.fc1_1 = nn.Sequential(nn.Linear(6, 70), nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(70, 100),
                                 nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(100, 6)).to(device)
        self.fc2_0 = nn.Sequential(nn.Linear(3, 70), nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(70, 100),
                                 nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(100, 3)).to(device)
        self.fc2_1 = nn.Sequential(nn.Linear(3, 70), nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(70, 100),
                                 nn.LeakyReLU(negative_slope=0.02, inplace=True), nn.Linear(100, 3)).to(device)
        self.fc11 = seidel2wavefront(device).to(device)
        self.fc12 = wavefront2psf(device).to(device)

    def forward(self, IS, H, color):
        if len(H.shape) == 0:
            H = H.unsqueeze(0).unsqueeze(0)
            BS = 1
        elif len(H.shape) == 1:
            BS = H.shape[0]
            H = H.unsqueeze(-1)
        torch.manual_seed(1)
        constant = torch.tensor([1.0]).float().unsqueeze(-1)
        con1 = constant.to(self.device).repeat(BS, 6)
        H1 = H.to(self.device).float().repeat(1, 6)
        H2 = H.to(self.device).float().repeat(1, 3)
        con2 = constant.to(self.device).repeat(BS, 3)
        coe1 = self.fc1_0(con1)*H1 + self.fc1_1(H1)*H1
        coe2 = self.fc2_0(con2)*H2 + self.fc2_1(H2)*H2
        coe = torch.cat([coe1,coe2], dim=1)
        wavefront = self.fc11(coe, IS, color, BS)
        psf = self.fc12(wavefront)
        return coe.float(), wavefront, psf.float()

    def init_weights(self):
        for param in [*self.fc1_0.parameters(), *self.fc1_1.parameters(), *self.fc2_0.parameters(), *self.fc2_1.parameters()]:
            if param.dim() > 1:  # Only apply to weights, not biases
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')


class shift_net(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(shift_net, self).__init__()
        self.device = device
        self.fc0 = nn.Sequential(nn.Linear(1, 40), nn.LeakyReLU(), nn.Linear(40, 20),
                                 nn.LeakyReLU(), nn.Linear(20, 4), nn.Tanh()).to(device)

    def forward(self, H):
        con = torch.tensor(H).to(self.device).unsqueeze(0)
        shift = 5 * self.fc0(con)
        return shift
