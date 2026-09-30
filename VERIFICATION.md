# Release verification — 2026-09-30

The seven included CPU runtime checks and checkpoint-restore checks passed. Using the installed Python 3.8 / PyTorch 1.12.1 CUDA environment, a real Spread Medium subset completed one pretraining update, eight online environment steps with three actor updates, checkpoint save/reload, and 20-episode evaluation calls. The pretraining-to-online zero boundary matched, checkpoint roundtrip was exact, and the pretrained source stayed unchanged. Real SMAC 3m, SMACv2 Terran 5-vs-5 and MA-MuJoCo 2Ant adapters each completed reset and five steps. The SMAC factory now has an environment-free regression check.

These are installation and short functional checks, not full-budget retraining or reproduction of paper scores. They do not certify every map, data split, historical checkpoint, optional rendering path, or inherited prototype. Simulator binaries/maps and datasets remain external requirements. Training seeds and evaluation seeds are distinct.
