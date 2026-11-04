

import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets import UnalignedDataset
from models import define_generator
from utils import save_image


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results/run")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=9)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--num_test", type=int, default=100)
    return p.parse_args()


def main():
    opt = get_args()
    device = "cuda" if (opt.gpu and torch.cuda.is_available()) else "cpu"

    os.makedirs(os.path.join(opt.results_dir, "fakeB"), exist_ok=True)  # A -> B
    os.makedirs(os.path.join(opt.results_dir, "fakeA"), exist_ok=True)  # B -> A

    dataset = UnalignedDataset(opt.dataroot, phase="test", crop_size=opt.crop_size, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    G = define_generator(3, 3, opt.ngf, opt.n_blocks, device)
    F = define_generator(3, 3, opt.ngf, opt.n_blocks, device)
    state = torch.load(opt.checkpoint, map_location=device)
    G.load_state_dict(state["G"])
    F.load_state_dict(state["F"])
    G.eval(); F.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= opt.num_test:
                break
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            fake_B = G(real_A)
            fake_A = F(real_B)

            name_A = os.path.splitext(os.path.basename(batch["path_A"][0]))[0]
            name_B = os.path.splitext(os.path.basename(batch["path_B"][0]))[0]
            save_image(fake_B[0], os.path.join(opt.results_dir, "fakeB", f"{name_A}_to_B.png"))
            save_image(fake_A[0], os.path.join(opt.results_dir, "fakeA", f"{name_B}_to_A.png"))

    print(f"[info] saved translated images to {opt.results_dir}")


if __name__ == "__main__":
    main()
