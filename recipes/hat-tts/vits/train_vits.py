import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.formatters import parse_ipa_hat_tts
from TTS.tts.models.vits import VitsAudioConfig, VitsArgs, CharactersConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.bin.compute_embeddings import compute_embeddings

SPEAKER_ENCODER_CHECKPOINT_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dataset_config = [
    BaseDatasetConfig(
        dataset_name=f"hat_tts_{name}",
        formatter="hat_tts",
        meta_file_train=f"{name}.json",
        path="/mnt/user_wayne/hat_tts",
        language="sixian" if "sixian" in name else "hailu",
    )
    for name in ["fa_sixian_concat_v2", "fa_hailu_concat_v2"]
]

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
            no_eval=False,
        )
    d_vector_files.append(embeddings_file)

audio_config = VitsAudioConfig(
    sample_rate=22050,
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

test_sentence = {
    "sixian": [
        ["tʰ+uŋ_11|h+ok_5"],
        ["p+et_2|pʰ+u_55"],
        ["v+uk_2|h+a_24"],
        ["kʰ+on_55|t+et_2 t+o_55"],
        [
            "ŋ+a_24|ɕ+in_24|s+aŋ_24 t+o_31|ɕ+in_24|s+am_24，h+i_55 tʰ+o-i_11|p+et_2 s+ɨ_55 k+e_55 k+u-et_2|k+a_24 im_24|ŋ+ok_5 tʰ+aŋ_24，tʰ+aŋ_24|im_24|ŋ+ok_5 f+i_55",
        ],
        [
            "h+ak_2|k+a_24 t͡sʰ+uk_5|kʰ+i-un_11 k+e_55 l+i-uk_2|t+o-i_24 i-un_55|tʰ+uŋ_55|f+i_55 f+i_55 it_2|t͡sʰ+ɨt_5 i-en_11 ɕ+i-uk_5 h+a_24|h+i_55，v+i_11 tʰ+o-i_11|v+an_11 k+e_55 tʰ+i_31|i-uk_2 s+ɨ_31 ɕ+i-a_31 h+a_24 tʰ+it_5|pʰ+i-et_5 k+e_55 it_2 i-ap_5"
        ],
        [
            "ɕ+in_24|t͡s+uk_2 t͡sʰ+a-i_55 kʰ+i-u_55|ŋ+i-en_11 i-u_24 ŋ+i_55|s+ɨp_5 s+am_24|ŋ+it_2 t͡sʰ+e-u_24|k+o_55 s+am_24|s+ɨp_5|l+i-uk_2 tʰ+u_55，m̩_11|t+i_24 n+on_24 f+a_55 k+e_55 pʰ+u_24|pʰ+i-en_55 tʰ+uŋ_11 m+a_31|k+e_55 i-u_24|k+u-an_24 h+e_55"
        ],
        [
            "tʰ+a-i_55|k+a_24 t+i-am_24 s+aŋ_24 t+en_31 ɕ+in_24|s+aŋ_24 k+i-e_31|s+ɨt_2，s+u_24|s+ɨn_11 m̩_11|h+a-u_55 m+o_11 f+an_55 h+o_31|s+ɨt_5 k+e_55 s+ɨ_11|t͡ɕ+i-et_2 o-i_55 ŋ+i-oŋ_55|k+at_2|s+at_2"
        ],
        [
            "k+u-i_24 tʰ+i-a-u_11 l+u_55 t+i-a-u_55|t+en_31 t͡sʰ+oŋ_11|t͡sʰ+oŋ_11 k+e_55 f+a_24|t+en_24，kʰ+i_11 kʰ+i-u_11 f+uŋ_24|tʰ+i-a-u_11|i_31|s+un_55，k+u-i_24 v+uk_2|h+a_24|ŋ+in_11 k+e_55 ɕ+im_24|ŋ+i-en_55，t͡ɕʰ+in_24|t͡ɕʰ+i-oŋ_55 f+a_24|t+en_24 h+a_24 s+e-u_24|n+on_24 k+e_55 k+oŋ_24|f+a_11"
        ],
    ],
    "hailu": [
        ["tʰ+uŋ_55|h+ok_2"],
        ["p+et_5|pʰ+u_33"],
        ["v+uk_5|h+a_53"],
        ["kʰ+on_11|t+et_5 t+o_11"],
        ["n+am_55|ŋ̩_24 pʰ+in_55|t+en_24 k+a-i_11 ʂ+i_55|tʰ+o-i_33，pʰ+i-aŋ_55|pʰ+i-aŋ_55 t͡s+o_11|t+et_5 ʂ+i-u_33 k+a-u_11|ʐ+uk_2"],
        ["k+a-i_24|k+i-et_5 s+am_53|t͡sʰ+on_53 m+o_55|l+ok_2|ʈ͡ʂʰ+ok_2 k+a-i_11 m+un_11|tʰ+i_55"],
        ["l+o_24|ŋ+in_55 kʰ+i-uŋ_33 t͡sʰ+on_53 k+a-i_11 ʈ͡ʂ+u_24|ʐ+a-u_11 t͡s+in_53|ʂ+in_55 ʐ+in_11|t+oŋ_53 s+ɨ_53|h+e_11 t͡s+o_11|pʰ+an_53"],
        ["l+i-en_55 p+ot_5|pʰ+i-aŋ_33 t+u_53 m+o_55|ŋ+in_55|t+i_53"],
        ["ʐ+i-u_53|k+i_53 f+u-i_33 tʰ+u_33 ŋ+i_55|t+e-u_53 t͡sʰ+am_53|k+a_53 s+in_53|t+en_53|p+an_24|t͡s+i-et_5 h+o_24|m+o_55"],
    ]
}

speakers = {
    "sixian": ["XF", "XM"],
    "hailu": ["HF", "HM"],
}

test_sentence

config = VitsConfig(
    audio=audio_config,
    model_args=vitsArgs,
    dashboard_logger="wandb",
    project_name="vits_hat_tts",
    run_name="vits_hat_tts",
    batch_size=64,
    eval_batch_size=128,
    batch_group_size=2,
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
    # use_phonemes=True,
    # phoneme_language="en-us",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters=[
            "11",
            "2",
            "24",
            "31",
            "33",
            "5",
            "53",
            "55",
            "a",
            "b",
            "d",
            "e",
            "f",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "s",
            "t",
            "u",
            "v",
            "w",
            "æ",
            "ŋ",
            "ɑ",
            "ɕ",
            "ə",
            "ɛ",
            "ɨ",
            "ɪ",
            "ʂ",
            "ʈ",
            "ʊ",
            "ʌ",
            "ʐ",
            "ʒ",
            "ʰ",
            "̩",
            "͡",
        ],
        punctuations="， ",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=dataset_config,
    cudnn_benchmark=False,
    max_audio_len=661500,  # file size
    test_sentences=[
        [sentence[0], speaker, None, dialect] for dialect in ["sixian", "hailu"] for sentence in test_sentence[dialect] for speaker in speakers[dialect]
    ],
    speaker_encoder_loss_alpha=9.0,
    grad_clip=[5.0, 5.0],
)

for test_sentence in config.test_sentences:
    test_sentence[0] = parse_ipa_hat_tts(test_sentence[0])

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
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
