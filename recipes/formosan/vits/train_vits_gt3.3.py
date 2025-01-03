import os

from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.characters import IPAPhonemes

SPEAKER_ENCODER_CHECKPOINT_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dataset_config = []

languages = [
    "ami",
    "sdq",
    "trv",
    "pwn",
    "tay",
    "bnn",
    "pyu",
    "dru",
    "xsy",
    "tsu",
    "tao",
    "ckv",
    "szy",
    "ssf",
    "sxr",
    "xnb",
]
dataset_config = []
for language in languages:
    dataset_config.append(
        BaseDatasetConfig(
            dataset_name=f"formosan_dict_{language}_train_gt3.3",
            formatter="formosan",
            meta_file_train=f"formosan_dict_{language}_concat.json",
            path="/home/wayne/Corpora/formosan_clean_denoised_train_gt3.3",
            language=language,
        )
    )
    dataset_config.append(
        BaseDatasetConfig(
            dataset_name=f"klokah_{language}_eval_gt3.3",
            formatter="formosan",
            meta_file_train=f"klokah_eval_{language}_concat.json",
            path="/home/wayne/Corpora/formosan_clean_denoised_train_gt3.3",
            language=language,
        )
    )
    if os.path.exists(
        "/home/wayne/Corpora/formosan_clean_denoised_train_gt3.3/ithuan_{language}_concat.json"
    ):
        dataset_config.append(
            BaseDatasetConfig(
                dataset_name=f"ithuan_{language}_train",
                formatter="formosan",
                meta_file_train=f"ithuan_{language}_concat.json",
                path="/home/wayne/Corpora/formosan_clean_denoised_train_gt3.3",
                language=language,
            )
        )


d_vector_files = []

speaker_emb_folder = os.path.join(output_path, "speaker_emb")
for dataset_conf in dataset_config:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(
        speaker_emb_folder, f"{dataset_conf.dataset_name}.pth"
    )
    if not os.path.isfile(embeddings_file):
        print(
            f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset"
        )
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=dataset_conf.meta_file_val == "",
        )
    d_vector_files.append(embeddings_file)

audio_config = VitsAudioConfig(
    sample_rate=24000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_language_embedding=True,
    embedded_language_dim=4,
    # use_speaker_embedding=True,
    num_layers_text_encoder=8,
    resblock_type_decoder="2",
    use_d_vector_file=True,
    d_vector_file=d_vector_files,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    use_speaker_encoder_as_loss=False,
)


# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config, eval_split=True, eval_split_size=0.015
)

test_sentence = eval_samples[:3]

config = VitsConfig(
    audio=audio_config,
    model_args=vitsArgs,
    dashboard_logger="wandb",
    project_name="yourtts_formosan",
    run_name="yourtts_formosan",
    batch_size=54,
    eval_batch_size=128,
    batch_group_size=4,
    num_loader_workers=16,
    num_eval_loader_workers=16,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=500,
    text_cleaner=None,
    use_speaker_embedding=False,
    use_language_embedding=True,
    target_loss="loss_1",
    save_best_after=50,
    characters=IPAPhonemes().to_config(),
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=dataset_config,
    cudnn_benchmark=False,
    max_audio_len=720000,  # file size
    test_sentences=[
        [sample["text"], None, None, sample["language"]] for sample in eval_samples[:3]
    ],
    speaker_encoder_loss_alpha=9.0,
    grad_clip=[5.0, 5.0],
    eval_split_size=0.01,
)

# init model
model = Vits.init_from_config(config)
# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
