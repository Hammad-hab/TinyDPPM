import torch, math

class NoiseScheduler:
    BetaStart = 1e-4
    BetaEnd=0.02

    def __init__(self, time, mode="linear", device="cpu") -> None:
        self.T = time
        self.s = 0.008
        if mode == "linear":
            self._betas: torch.Tensor = torch.linspace(NoiseScheduler.BetaStart, NoiseScheduler.BetaEnd, self.T)
            self._alphas = 1.0 - self._betas
            self._alphab = torch.cumprod(self._alphas)

        elif mode == 'cosine':
            self._tms: torch.Tensor = torch.linspace(0, self.T, self.T+1)
            self._num = torch.square(torch.cos(((self._tms/self.T + self.s)/(1+self.s))*(torch.pi/2)))
            self._denom = math.pow(math.cos((self.s/(1+self.s))*(torch.pi/2)), 2)

            self._alphab = self._num/self._denom
            self._alphas = (self._alphab[1:]/self._alphab[:-1]);
            self._betas = 1-self._alphas

        else:
            raise ValueError(f"Unknown mode {mode}")

        self.to(device)

    def to(self, device):
        self._betas.to(device)
        self._alphas.to(device)
        self._alphab.to(device)

    def getAlpha(self, t):
         return self._alphas[t]

    def getBeta(self, t):
        return self._betas[t]

    def get(self, t):
        return (self.getBeta(t), self.getAlpha(t))


if __name__ == "__main__":
    ns = NoiseScheduler(1000, "cosine")
    assert (len(ns._betas)) == (len(ns._alphas)), "Lengths of betas and alphas must be the same"
    assert (1-ns._alphab[0].detach().item())<1e-6, "zeroth element of alpha-bar must be 1"
