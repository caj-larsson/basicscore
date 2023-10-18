import json
import argparse
from typing import List
from tqdm import tqdm
import json
from pathlib import Path
import argparse
import appdirs
import os.path
from collections import ChainMap
from basicscore import BasicScore, Config, from_config
from basicscore.cli.html import render_file_html


def parseargs():
    parser = argparse.ArgumentParser(
        description="BasicScore a file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file"
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        default=None,
        help="File to score"
    )
    parser.add_argument(
        "--token-output",
        type=Path,
        default=None,
        help="Output json data with all token probabilities"
    )
    parser.add_argument("--gpu-layers", type=bool, default=0, help="Layers on GPU")
    parser.add_argument("--html-output", type=Path, default=None, help="Write html report")
    parser.add_argument("--config", type=Path, default=None, help="Use specific config file")
    parser.add_argument("--context-prompt", type=int, default=20, help="Backtrack context characters in prompt")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads to use")
    parser.add_argument("--score", type=str, default="avgprob", help="Score function [""]")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context window length")
    return parser.parse_args()


def buildConfig(args):
    dirs = appdirs.AppDirs(appname="basicscore")
    config_paths = [
        args.config,
        Path(os.getcwd(), ".basicscore.json"),
        Path(dirs.user_config_dir, "config.json")
    ]
    config_dict = {}
    for path in config_paths:
        if path is None:
            continue
        try:
            with open(path) as f:
                config_dict = json.load(f)
                break
        except Exception:
            continue

    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    return Config(**ChainMap(args_dict, config_dict))


def main():
    args = parseargs()
    config = buildConfig(args)

    if args.files is None:
        print("Please add --files")
        exit(1)

    progress_bar = tqdm(unit_scale=True)
    def update():
        progress_bar.update(1)

    bs = from_config(config, update)

    file_scores = {}
    file_token_probs = {}

    for filename in args.files:
        with open(filename, "rb") as f:
            s = bs.score_unit(filename,f.read(), update)
            file_token_probs[str(filename)] = bs.probs
            file_scores[str(filename)] = s

    if args.html_output is not None:
        with open(args.html_output, "w") as f:
            files = [{
                "filename": filename,
                "score": file_scores[filename],
                "tokens": file_token_probs[filename]
            } for filename in file_scores.keys()]
            f.write(render_file_html(files))

    progress_bar.close()
    for filename, score in file_scores.items():
        print(f"{filename}: {score}")

    if args.token_output:
        with open(args.token_output, "w") as f:
            json.dump({"scores": file_scores, "tokens": file_token_probs}, f)
