import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletDWT(nn.Module):
    def __init__(self, wave='haar', J=1):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave)

    def forward(self, x):
        yl, yh = self.dwt(x)  
        yh = yh[0]             

        LL = yl                
        LH = yh[:, :, 0]      
        HL = yh[:, :, 1]       
        HH = yh[:, :, 2]      

        return LL, (LH, HL, HH)


class IDWT(nn.Module):

    def __init__(self, wave='haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave)

    def forward(self, LL, LH, HL, HH):

        yh = torch.stack([LH, HL, HH], dim=2)  # -> (B, C, 3, H/2, W/2)
        yl = LL  

        x_rec = self.idwt((yl, [yh]))
        return x_rec

if __name__ == "__main__":
    # model = WaveletDWT()
    # x = torch.randn(1, 3, 128, 128)
    # LL, (LH, HL, HH) = model(x)

    dwt = DWTForward(J=1, wave='haar')
    idwt = IDWT(wave='haar')

    x = torch.randn(1, 3, 128, 128)
    yl, yh = dwt(x)
    yh = yh[0]                # (B, C, 3, H/2, W/2)
    LL = yl
    LH, HL, HH = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]

    x_rec = idwt(LL, LH, HL, HH)

    print("Original shape:", x.shape)
    print("Reconstructed shape:", x_rec.shape)
    print("Reconstruction error:", torch.mean((x - x_rec)**2).item())
