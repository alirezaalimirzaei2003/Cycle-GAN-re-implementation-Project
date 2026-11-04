
import argparse
import itertools
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import UnalignedDataset
from models import define_generator, define_discriminator, GANLoss
from utils import ImagePool, linear_lr_lambda, save_image


def get_args():
    p = argparse.ArgumentParser(description="CycleGAN training (paper-faithful reproduction)")
    p.add_argument("--dataroot", type=str, required=True, help="path with trainA/ trainB/ subfolders")
    p.add_argument("--name", type=str, default="cyclegan_run")
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    p.add_argument("--load_size", type=int, default=286)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=9, help="9 for 256x256, 6 for 128x128 (paper default)")
    p.add_argument("--n_epochs", type=int, default=100, help="epochs at constant (initial) LR")
    p.add_argument("--n_epochs_decay", type=int, default=100, help="epochs to linearly decay LR to 0")
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_identity", type=float, default=0.5,
                    help="weight of identity loss relative to lambda_cyc; set 0 to disable")
    p.add_argument("--pool_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_epoch_freq", type=int, default=5)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--resume", action="store_true",
                    help="auto-resume from checkpoints/<name>/latest.pth if it exists "
                         "(restores model weights, optimizer state, and the correct LR "
                         "schedule position) -- use this to pause/continue training across runs")
    return p.parse_args()


def main():
    opt = get_args()
    device = "cuda" if (opt.gpu and torch.cuda.is_available()) else "cpu"
    print(f"[info] using device: {device}")

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    
    dataset = UnalignedDataset(opt.dataroot, phase="train", load_size=opt.load_size,
                                crop_size=opt.crop_size, is_train=True)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                         num_workers=opt.num_workers, drop_last=True)
    print(f"[info] dataset size: {len(dataset)} (A={dataset.size_A}, B={dataset.size_B})")

    
    G = define_generator(3, 3, opt.ngf, opt.n_blocks, device)   # A -> B
    F = define_generator(3, 3, opt.ngf, opt.n_blocks, device)   # B -> A
    D_A = define_discriminator(3, opt.ndf, 3, device)           # real A vs F(B)
    D_B = define_discriminator(3, opt.ndf, 3, device)           # real B vs G(A)

    criterion_gan = GANLoss().to(device)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    opt_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()),
                              lr=opt.lr, betas=(opt.beta1, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                              lr=opt.lr, betas=(opt.beta1, 0.999))

    lr_lambda = lambda ep: linear_lr_lambda(ep, opt.n_epochs, opt.n_epochs_decay, epoch_count=1)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    pool_fake_A = ImagePool(opt.pool_size)
    pool_fake_B = ImagePool(opt.pool_size)

    total_epochs = opt.n_epochs + opt.n_epochs_decay
    start_epoch = 1

    
    latest_ckpt = os.path.join(save_dir, "latest.pth")
    if opt.resume and os.path.exists(latest_ckpt):
        state = torch.load(latest_ckpt, map_location=device)
        G.load_state_dict(state["G"]); F.load_state_dict(state["F"])
        D_A.load_state_dict(state["D_A"]); D_B.load_state_dict(state["D_B"])
        opt_G.load_state_dict(state["opt_G"])
        opt_D.load_state_dict(state["opt_D"])
        completed_epochs = state["epoch"]
        for _ in range(completed_epochs):
            sched_G.step()
            sched_D.step()
        start_epoch = completed_epochs + 1
        print(f"[info] resumed from {latest_ckpt} -> continuing at epoch {start_epoch} "
              f"(lr={opt_G.param_groups[0]['lr']:.6f})")
    elif opt.resume:
        print(f"[info] --resume given but no checkpoint found at {latest_ckpt}, starting fresh")

    
    print("[info] press Ctrl+C any time to stop -- latest.pth is only ever "
          "written at the END of a completed epoch, so a checkpoint is "
          "never left half-written; just re-run with --resume to continue.")
    try:
        _run_training(opt, device, save_dir, sample_dir, loader,
                       G, F, D_A, D_B, criterion_gan, criterion_cycle, criterion_identity,
                       opt_G, opt_D, sched_G, sched_D, pool_fake_A, pool_fake_B,
                       total_epochs, start_epoch)
    except KeyboardInterrupt:
        print("\n[info] interrupted by user. Progress up to the last completed "
              "epoch is already saved in latest.pth. Re-run with --resume to continue.")


def _run_training(opt, device, save_dir, sample_dir, loader,
                   G, F, D_A, D_B, criterion_gan, criterion_cycle, criterion_identity,
                   opt_G, opt_D, sched_G, sched_D, pool_fake_A, pool_fake_B,
                   total_epochs, start_epoch):
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        for i, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # ---------------- Generators G and F ---------------- #
            for net in (D_A, D_B):
                for p in net.parameters():
                    p.requires_grad = False
            opt_G.zero_grad()

            # Identity loss
            loss_idt = torch.tensor(0.0, device=device)
            if opt.lambda_identity > 0:
                idt_B = G(real_B)
                idt_A = F(real_A)
                loss_idt = (criterion_identity(idt_B, real_B) + criterion_identity(idt_A, real_A)) \
                    * opt.lambda_cyc * opt.lambda_identity

            fake_B = G(real_A)          # G: A -> B
            fake_A = F(real_B)          # F: B -> A

            # Adversarial loss: L_GAN
            loss_gan_G = criterion_gan(D_B(fake_B), True)
            loss_gan_F = criterion_gan(D_A(fake_A), True)

            # Cycle-consistency loss 
            rec_A = F(fake_B)
            rec_B = G(fake_A)
            loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * opt.lambda_cyc

            loss_G_total = loss_gan_G + loss_gan_F + loss_cycle + loss_idt
            loss_G_total.backward()
            opt_G.step()

            # ---------------- Discriminators Dx and Dy ---------------- #
            for net in (D_A, D_B):
                for p in net.parameters():
                    p.requires_grad = True
            opt_D.zero_grad()

            fake_A_ = pool_fake_A.query(fake_A)
            fake_B_ = pool_fake_B.query(fake_B)

            loss_D_A = 0.5 * (criterion_gan(D_A(real_A), True) + criterion_gan(D_A(fake_A_.detach()), False))
            loss_D_B = 0.5 * (criterion_gan(D_B(real_B), True) + criterion_gan(D_B(fake_B_.detach()), False))
            (loss_D_A + loss_D_B).backward()
            opt_D.step()

            if i % opt.print_freq == 0:
                print(f"epoch {epoch}/{total_epochs} iter {i}/{len(loader)} "
                      f"G:{loss_G_total.item():.3f} (gan {loss_gan_G.item()+loss_gan_F.item():.3f}, "
                      f"cyc {loss_cycle.item():.3f}, idt {loss_idt.item():.3f}) "
                      f"D_A:{loss_D_A.item():.3f} D_B:{loss_D_B.item():.3f}")

        sched_G.step()
        sched_D.step()
        print(f"[epoch {epoch}] done in {time.time()-t0:.1f}s, lr={opt_G.param_groups[0]['lr']:.6f}")

        # Save qualitative samples + checkpoint
        save_image(real_A[0], os.path.join(sample_dir, f"epoch{epoch}_realA.png"))
        save_image(fake_B[0], os.path.join(sample_dir, f"epoch{epoch}_fakeB.png"))
        save_image(real_B[0], os.path.join(sample_dir, f"epoch{epoch}_realB.png"))
        save_image(fake_A[0], os.path.join(sample_dir, f"epoch{epoch}_fakeA.png"))

        # "latest.pth" is saved after EVERY epoch 
        full_state = {"G": G.state_dict(), "F": F.state_dict(),
                       "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                       "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                       "epoch": epoch}
        torch.save(full_state, os.path.join(save_dir, "latest.pth"))
        print(f"[info] latest.pth updated (epoch {epoch}) -- safe to pause now")

        
        if epoch % opt.save_epoch_freq == 0 or epoch == total_epochs:
            torch.save({"G": G.state_dict(), "F": F.state_dict(),
                        "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                        "epoch": epoch}, os.path.join(save_dir, f"epoch_{epoch}.pth"))
            print(f"[info] named checkpoint saved at epoch {epoch}")


if __name__ == "__main__":
    main()
