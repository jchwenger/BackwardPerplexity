import os
import rjsmin
import requests
import argparse
from tqdm import tqdm
from datasets import load_dataset


def query(url):
    response = requests.get(url, headers=headers)
    return response.json()


def collect_dataset(
    args,
    # out_txt_path, language, minified, api_token, file_limit
):

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{args.language}",
        split="train",
        streaming=True,
        token=args.api_token,
    )

    with open(out_txt_path, "w") as o:
        for i, row in enumerate(tqdm(ds, total=args.file_limit - 1)):
            content = row["content"]
            if args.minified and args.language == "javascript":
                content = rjsmin.jsmin(content)
            if args.no_comments:
                raise NotImplementedError
            # Write content followed by a newline character to separate each entry
            o.write(content + "\n")
            if i == args.file_limit:  # Stop after collecting N rows
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Collecting code snippets from specific languages in The
        Stack by BigCode (https://huggingface.co/datasets/bigcode/the-stack)
        into a single .txt file"""
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default="javascript",
        help="""
        The programming language. See the list here:
        https://huggingface.co/datasets/bigcode/the-stack/blob/main/programming-languages.json
        """,
    )

    parser.add_argument(
        "--directory", "-d",
        type=str,
        default="code_dataset",
        help="""
        The directory to use. Defaults to 'code_dataset.{language}'. Specify
        only the first part, e.g.: 'custom_dataset' with Python will yield
        'custom_dataset.python'
        """,
    )

    parser.add_argument(
        "--api_token", "-t",
        type=str,
        required=True,
        help="""
        The Huggingface api token. Required to access The Stack. See here:
        https://huggingface.co/settings/tokens""",
    )

    parser.add_argument(
        "--file_limit", "-L",
        default=5000,
        type=int,
        help="""The maximum number of files to handle. Defaults to 5000.""",
    )

    cleaning_group = parser.add_mutually_exclusive_group()

    cleaning_group.add_argument(
        "--minified", "-m",
        action="store_true",
        help="""
        Minifies the code (only for javascript), using rjsmin.
        https://pypi.org/project/rjsmin/
        """,
    )

    cleaning_group.add_argument(
        "--no_comments", "-n",
        action="store_true",
        help="""
        Removes comments, using Comment Parser.
        https://pypi.org/project/comment-parser/
        """
    )

    args = parser.parse_args()

    args.directory += f".{args.language}"

    if args.minified:
        args.directory += "_minified"

    if args.no_comments:
        args.directory += "_no_comments"

    if not os.path.isdir(args.directory):
        print(f"{args.directory} folder not found, creating.")
        os.mkdir(args.directory)

    out_txt_path = os.path.join(args.directory, f"{args.language}.txt")

    collect_dataset(args)
