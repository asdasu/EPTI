import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def main():
    crypten.init()
    device = "cuda"
    runs = 10
    num_vecs = 1024  


    softmax_time, softmax_bytes, softmax_rounds = {}, {}, {}


    torch.manual_seed(0)

    for softmax_l in [32, 64, 128, 256]:

        crypten.reset_communication_stats()
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        max_err = 0.0
        avg_err_sum = 0.0


        for _ in range(runs):

            x_plain = torch.randn(num_vecs, softmax_l, device=device)


            y_ref = torch.softmax(x_plain, dim=-1)

            x_enc = crypten.cryptensor(x_plain, device=device)

            y_enc = x_enc.softmax(-1)
            y_dec = y_enc.get_plain_text()  

            diff = (y_dec - y_ref).abs()  

            avg_err_run = diff.mean().item()

            avg_err_sum += avg_err_run

        if device == "cuda":
            torch.cuda.synchronize()

        softmax_time[softmax_l] = time.time() - start_time
        stats = crypten.get_communication_stats()
        softmax_bytes[softmax_l] = stats["bytes"]
        softmax_rounds[softmax_l] = stats["rounds"]



    if crypten.comm.get().get_rank() == 0:
        for softmax_l in [32, 64, 128, 256]:
            print(
                f"l={softmax_l}, num_vecs={num_vecs} "
                f"time: {softmax_time[softmax_l] / runs:.4f}s, "
                f"bytes: {softmax_bytes[softmax_l] / 1048576 / runs:.4f} MB, "
                f"rounds: {softmax_rounds[softmax_l] / runs:.0f}"
            )
            
if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()