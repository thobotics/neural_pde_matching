import torch


class TrajectoryMetrics:
    @staticmethod
    def cummulative_mse(mse_sample_wise: torch.Tensor):
        mse = mse_sample_wise.sum(-1)
        loss = torch.cumsum(torch.mean(mse, dim=1), dim=0)
        loss /= torch.arange(1, len(loss) + 1).to(loss.device)

        return loss

    @staticmethod
    def nth_mse(mse_sample_wise: torch.Tensor, n: int):
        mse = mse_sample_wise.sum(-1)
        loss = torch.mean(mse[n])

        return loss
