import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def main():

    from crypten.config import cfg
    cfg.load_config("/root/SHAFT/configs/default.yaml")
    print(">>> GELU using method:", cfg.functions.gelu_method)
    # ================================

    crypten.init()
    device = "cuda"
    runs = 10

    gelu_time, gelu_bytes, gelu_rounds = {}, {}, {}
    approximate = "none"  

    x = torch.arange(-5, 5, 0.001)

    y_original = torch.nn.functional.gelu(x, approximate=approximate)

    y_actual = crypten.cryptensor(x).gelu(approximate=approximate).get_plain_text()

    err = (y_original - y_actual).abs()

    max_err_all = err.max()
    avg_err_all = err.mean()

    mask_mid = (x >= -2) & (x <= 2)
    err_mid = err[mask_mid]

    max_err_mid = err_mid.max()
    avg_err_mid = err_mid.mean()
    for gelu_size in [(128, 4096), (256, 8192)]:
        gelu_in = crypten.cryptensor(torch.zeros(gelu_size), device=device)

        crypten.reset_communication_stats()
        start_time = time.time()

        for _ in range(runs):
            gelu_in.gelu(approximate=approximate)

        gelu_time[gelu_size[1]] = time.time() - start_time
        stats = crypten.get_communication_stats()
        gelu_bytes[gelu_size[1]] = stats["bytes"]
        gelu_rounds[gelu_size[1]] = stats["rounds"]

    if crypten.comm.get().get_rank() == 0:
        print(f"[-5,5]  max error: {max_err_all:.6f}, avg error: {avg_err_all:.6f}")
        print(f"[-2,2]  max error: {max_err_mid:.6f}, avg error: {avg_err_mid:.6f}")

        for gelu_size in [(128, 4096), (256, 8192)]:
            print(f"({gelu_size[0]}, {gelu_size[1]}) "
                  f"time: {gelu_time[gelu_size[1]] / runs:.4f}s, "
                  f"bytes: {gelu_bytes[gelu_size[1]] / 1048576 / runs:.0f} MB, "
                  f"rounds: {gelu_rounds[gelu_size[1]] / runs:.0f}")


if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()