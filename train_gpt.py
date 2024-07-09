"""
    Training script for GPT-like models. Use this along with the output of gen_run !

    Usage : python train_gpt.py <path_to_json_config_file> -d <device> -t <tokenizer_path> -p <project_name> -s

    Example : python train_gpt.py TrainParams/params.json -d cuda:0 -t fr -p BackPerplexityResilient -s

    Note : uses wandb, so you need to have a wandb account and be logged in.
"""

from train import train
import argparse



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Starts training of Predictor model given a JSON config file.")
    parser.add_argument("file_location", help="Path to the JSON config file. Relative to where you launch the script from.")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("-t", "--tokenizer_path", type=str,help="Path for the tokenizer to use (only used for logging snippets). Relative to the train_script folder.")
    parser.add_argument("-p", "--project_name", help="Name of the project to log to. Default is 'BackPerplexityResilient'")
    parser.add_argument("-s", "--no_step_pickup", action="store_false", help="If set, train steps_to_train steps more. Otherwise, will train UP TO steps_to_train TOTAL steps.")  
    args = parser.parse_args()

    if(args.project_name is None):
        project_name = 'CodePerplexity'
    else:
        project_name = args.project_name

    train(model_name='gpt',file_location=args.file_location, device=args.device, tokenizer_path=args.tokenizer_path, 
          project_name=project_name, step_pickup=args.no_step_pickup)