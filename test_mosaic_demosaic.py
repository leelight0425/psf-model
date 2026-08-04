"""Prove demosaic bugs with B=2 batch input."""
import torch
import torch.nn as nn

# ── Old (buggy) demosaic ──
def demosaic_old(bayer_images):
    def S2D(x):
        bs=2; N,C,H,W=x.size(); x=x.view(N,C,H//bs,bs,W//bs,bs)
        x=x.permute(0,3,5,1,2,4).contiguous(); return x.view(N,C*4,H//bs,W//bs)
    def D2S(x):
        bs=2; N,C,H,W=x.size(); x=x.view(N,bs,bs,C//4,H,W)
        x=x.permute(0,3,4,1,5,2).contiguous(); return x.view(N,C//4,H*bs,W*bs)
    if len(bayer_images.shape) == 3:
        bayer_images = bayer_images.unsqueeze(0)
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    B,H,W,C = bayer_images.shape
    up = nn.Upsample(size=[H*2, W*2], mode='bilinear', align_corners=False)
    red = up(bayer_images[..., 0:1].permute(0,3,1,2)).permute(0,2,3,1)

    gr = bayer_images[..., 1:2]
    gr = torch.flip(gr, dims=[1]); gr = up(gr.permute(0,3,1,2)).permute(0,2,3,1)
    gr = torch.flip(gr, dims=[1]); gr = S2D(gr.permute(0,3,1,2)).permute(0,2,3,1)

    gb = bayer_images[..., 2:3]
    gb = torch.flip(gb, dims=[0]); gb = up(gb.permute(0,3,1,2)).permute(0,2,3,1)
    gb = torch.flip(gb, dims=[0]); gb = S2D(gb.permute(0,3,1,2)).permute(0,2,3,1)

    g_r  = (gr[...,0] + gb[...,0]) / 2
    g_gr = gr[...,1]; g_gb = gb[...,2]
    g_b  = (gr[...,3] + gb[...,3]) / 2
    green = D2S(torch.stack([g_r,g_gr,g_gb,g_b], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)

    blue = bayer_images[..., 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = up(blue.permute(0,3,1,2)).permute(0,2,3,1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb = torch.cat([red,green,blue], dim=-1)
    return rgb.permute(0,3,1,2)

# ── New (fixed) demosaic ──
def demosaic_new(bayer_images):
    def S2D(x):
        bs=2; N,C,H,W=x.size(); x=x.view(N,C,H//bs,bs,W//bs,bs)
        x=x.permute(0,3,5,1,2,4).contiguous(); return x.view(N,C*4,H//bs,W//bs)
    def D2S(x):
        bs=2; N,C,H,W=x.size(); x=x.view(N,bs,bs,C//4,H,W)
        x=x.permute(0,3,4,1,5,2).contiguous(); return x.view(N,C//4,H*bs,W*bs)
    if len(bayer_images.shape) == 3:
        bayer_images = bayer_images.unsqueeze(0)
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    B,H,W,C = bayer_images.shape
    up = nn.Upsample(size=[H*2, W*2], mode='bilinear', align_corners=False)
    red = up(bayer_images[..., 0:1].permute(0,3,1,2)).permute(0,2,3,1)

    gr = bayer_images[..., 1:2]
    gr = torch.flip(gr, dims=[2]); gr = up(gr.permute(0,3,1,2)).permute(0,2,3,1)  # FIXED
    gr = torch.flip(gr, dims=[2]); gr = S2D(gr.permute(0,3,1,2)).permute(0,2,3,1)  # FIXED

    gb = bayer_images[..., 2:3]
    gb = torch.flip(gb, dims=[1]); gb = up(gb.permute(0,3,1,2)).permute(0,2,3,1)  # FIXED
    gb = torch.flip(gb, dims=[1]); gb = S2D(gb.permute(0,3,1,2)).permute(0,2,3,1)  # FIXED

    g_r  = (gr[...,0] + gb[...,0]) / 2
    g_gr = gr[...,1]; g_gb = gb[...,2]
    g_b  = (gr[...,3] + gb[...,3]) / 2
    green = D2S(torch.stack([g_r,g_gr,g_gb,g_b], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)

    blue = bayer_images[..., 3:4]
    blue = torch.flip(blue, dims=[1,2]); blue = up(blue.permute(0,3,1,2)).permute(0,2,3,1)  # FIXED
    blue = torch.flip(blue, dims=[1,2])  # FIXED

    rgb = torch.cat([red,green,blue], dim=-1)
    return rgb.permute(0,3,1,2)

def mosaic(image):  # supports BHWC or HWC (no batch) -> Bx4xH/2xW/2
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    B, H, W, C = image.shape
    out = torch.zeros(B, 4, H//2, W//2)
    out[:,0] = image[:, 0::2, 0::2, 0]  # R
    out[:,1] = image[:, 0::2, 1::2, 1]  # Gr
    out[:,2] = image[:, 1::2, 0::2, 1]  # Gb
    out[:,3] = image[:, 1::2, 1::2, 2]  # B
    return out


if __name__ == "__main__":
    # B=2: two DIFFERENT images in the batch
    H, W = 16, 16
    img1 = torch.zeros(H, W, 3)
    img1[:,:,0] = 1.0  # R=1
    img1[:,:,1] = 0.5  # G=0.5
    img1[:,:,2] = 0.0  # B=0

    img2 = torch.zeros(H, W, 3)
    img2[:,:,0] = 0.0  # R=0
    img2[:,:,1] = 0.5  # G=0.5
    img2[:,:,2] = 1.0  # B=1

    batch = torch.stack([img1, img2], dim=0)  # (2, H, W, 3)
    bay = mosaic(batch)
    print(f"Bayer shape: {bay.shape}")  # (2, 4, 8, 8)

    rec_old = demosaic_old(bay)
    rec_new = demosaic_new(bay)

    print("\n=== B=2 Batch Test ===")
    print(f"Image 0 GT: R=1.0, G=0.5, B=0.0")
    print(f"Image 1 GT: R=0.0, G=0.5, B=1.0")

    for b in range(2):
        print(f"\n--- Batch item {b} ---")
        for i, name in enumerate(["R", "G", "B"]):
            old_val = rec_old[b, i].mean().item()
            new_val = rec_new[b, i].mean().item()
            gt_val = batch[b, :, :, i].mean().item()
            print(f"  {name}: GT={gt_val:.4f}  Old(buggy)={old_val:.4f}  New(fixed)={new_val:.4f}")

    # Key test: did old code mix channels across batch?
    print("\n=== Cross-batch contamination check ===")
    # In old code, dims=[0] flip on green_blue would swap batch items
    # After flip-back, batch items should be swapped back... but with B=2,
    # a double flip on dims=[0] would restore order
    # Actually, the real issue is that the flip is on the wrong axis
    # Let me check if the channel values are correct
    old_r0 = rec_old[0, 0].mean().item()
    new_r0 = rec_new[0, 0].mean().item()
    old_r1 = rec_old[1, 0].mean().item()
    new_r1 = rec_new[1, 0].mean().item()
    print(f"  Img0 R:  expected=1.0, old={old_r0:.4f}, new={new_r0:.4f}")
    print(f"  Img1 R:  expected=0.0, old={old_r1:.4f}, new={new_r1:.4f}")
    old_b0 = rec_old[0, 2].mean().item()
    new_b0 = rec_new[0, 2].mean().item()
    old_b1 = rec_old[1, 2].mean().item()
    new_b1 = rec_new[1, 2].mean().item()
    print(f"  Img0 B:  expected=0.0, old={old_b0:.4f}, new={new_b0:.4f}")
    print(f"  Img1 B:  expected=1.0, old={old_b1:.4f}, new={new_b1:.4f}")
