import torch
from TTS.tts.utils.speakers import SpeakerManager
from TTS.config import load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", help="Output file path", required=True)
    parser.add_argument("-m", "--model_folder", help="Model folder path", required=True)
    args = parser.parse_args()
    output_path = args.output_path
    model_folder = args.model_folder

    config = load_config(model_folder+"/config.json")
    speaker_manager = SpeakerManager.init_from_config(config)

    speaker_emb_dict = {}
    for speaker in speaker_manager.get_speakers():
        speaker_emb_dict[speaker] = {
            "name": speaker,
            "embedding": speaker_manager.get_mean_embedding(speaker).tolist()
        }


    torch.save(speaker_emb_dict, output_path)