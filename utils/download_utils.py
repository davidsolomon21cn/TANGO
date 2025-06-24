import os
from huggingface_hub import snapshot_download, hf_hub_download


def download_files_from_repo():
    # check the last ckpts are downloaded
    repo_id = "H-Liu1997/TANGO"
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    last_ckpt_path = os.path.join(local_dir, "datasets/cached_graph/youtube_test/speaker9.pkl")
    if os.path.exists(last_ckpt_path):
        return
    else:
        snapshot_download(
            repo_id="H-Liu1997/tango_cached_utils", local_dir=local_dir, repo_type="dataset", ignore_patterns="datasets/cached_graph/*", force_download=True
        )
        snapshot_download(
            repo_id="H-Liu1997/tango_cached_utils", local_dir=local_dir, repo_type="dataset", allow_patterns="datasets/cached_graph/youtube_test/speaker9.pkl", force_download=True
        )
    print("Downloaded all the necessary files from the repo.")
