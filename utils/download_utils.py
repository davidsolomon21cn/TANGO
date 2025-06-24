import os
from huggingface_hub import snapshot_download, hf_hub_download


def download_files_from_repo():
    # check the last ckpts are downloaded
    repo_id = "H-Liu1997/TANGO"
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    last_ckpt_path = os.path.join(local_dir, "SMPLer-X/pretrained_models/smpler_x_s32.pth.tar")
    if os.path.exists(last_ckpt_path):
        return
    else:
        # separate the download of the files for better debugging
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="space", allow_patterns="frame-interpolation-pytorch/*.pt", force_download=True
        )
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="space", allow_patterns="Wav2Lip/checkpoints/*.pth", force_download=True
        )
        hf_hub_download(
            repo_id="H-Liu1997/cached_ckpts",
            filename="ckpt.pth",
            repo_type="dataset",
            local_dir=os.path.join(local_dir, "datasets/cached_ckpts"),
            force_download=True,
            local_dir_use_symlinks=False,
        )
        hf_hub_download(
            repo_id="H-Liu1997/cached_graph",
            filename="youtube_test/speaker1.pkl",
            repo_type="dataset",
            local_dir=os.path.join(local_dir, "datasets/cached_graph"),
            force_download=True,
            local_dir_use_symlinks=False,
        )
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="space", allow_patterns="emage/smplx_models/smplx/*", force_download=True
        )
        snapshot_download(
            repo_id="caizhongang/SMPLer-X",
            local_dir=os.path.join(local_dir, "./SMPLer-X"),
            repo_type="space",
            ignore_patterns="pretrained_models/*",
            force_download=True,
        )
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="space", allow_patterns="SMPLer-X/pretrained_models/*", force_download=True
        )
    print("Downloaded all the necessary files from the repo.")
