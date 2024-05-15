import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.formatters import parse_ipa_hat_tts
from TTS.tts.models.vits import VitsAudioConfig, CharactersConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor



output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = [
    BaseDatasetConfig(
        formatter="hat_tts",
        meta_file_train=f"{dialect}.json",
        path="/mnt/user_wayne/hat_tts",
        language=dialect,
        ignored_speakers=["XM"]
    )
    for dialect in ["fa_sixian_concat", "fa_sixian"]
]
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# vitsArgs = VitsArgs(
#     # use_language_embedding=True,
#     # embedded_language_dim=4,
#     # use_speaker_embedding=True,
#     num_layers_text_encoder=10,
#     # resblock_type_decoder="2"
# )

config = VitsConfig(
    audio=audio_config,
    # model_args=vitsArgs,
    dashboard_logger="wandb",
    project_name="vits_hat_tts",
    run_name="vits_hat_tts",
    batch_size=64, #32
    eval_batch_size=128,
    batch_group_size=8,
    num_loader_workers=16,
    num_eval_loader_workers=16,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner=None,
    use_speaker_embedding=False,
    use_language_embedding=False,
    # use_phonemes=True,
    # phoneme_language="en-us",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters=['11', '2', '24', '31', '5', '53', '55', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'æ', 'ŋ', 'ɑ', 'ɕ', 'ɛ', 'ɨ', 'ɪ', 'ʊ', 'ʒ', 'ʰ', '̩', '͡'],
        punctuations=" ，",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    compute_input_seq_cache=True,
    print_step=50,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=dataset_config,
    cudnn_benchmark=False,
    max_audio_len=1440000,  # file size
    test_sentences=[
        [
            "tʰ+uŋ_11|h+ok_5"
        ],
        [
            "p+et_2|pʰ+u_55"
        ],
        [
            "v+uk_2|h+a_24"
        ],
        [
            "kʰ+on_55|t+et_2 t+o_55"
        ],
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
        ]
    ],
)

for test_sentence in config.test_sentences:
    test_sentence[0] = parse_ipa_hat_tts(test_sentence[0])

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

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

# speaker_manager = SpeakerManager()
# speaker_manager.set_ids_from_data(train_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# init model
model = Vits(
    config,
    ap,
    tokenizer,
    # speaker_manager=speaker_manager,
    # language_manager=language_manager,
)

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
