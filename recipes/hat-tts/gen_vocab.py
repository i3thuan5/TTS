import glob
import re
import json

from TTS.tts.datasets.formatters import parse_ipa_hat_tts, parse_ipa_hat_tts_initial_final

# DATASET = ["/mnt/user_wayne/hat_tts/", "/mnt/user_wayne/hac_vocab/", "/mnt/user_wayne/moe_hakkadict/"]
DATASET = ["/mnt/user_wayne/hat_tts/"]

# def parse_ipa_hat_tts(ipa: str, initial_final=False):
#     text = []
#     ipa_list = re.split(r"(?<![， -])(?=[， -])|(?<=[， -])(?![， -])",ipa)
#     # tone as a separate token
#     for phoneme_with_tone in ipa_list:
#         if phoneme_with_tone ==" ":
#             text.append(phoneme_with_tone)
#             continue
#         elif phoneme_with_tone == "，":
#             text.extend(" ， ")
#             continue
#         elif phoneme_with_tone == "-": # use " " split 詞 (or use " " to split 字)
#             continue

#         split_phoneme_and_tone = phoneme_with_tone.split("_")

#         if len(split_phoneme_and_tone) == 2:
#             phoneme, tone = split_phoneme_and_tone
#             if initial_final:
#                 text.append(phoneme)
#             else:
#                 text.extend(phoneme) 

#             text.append(tone)
#         else:
#             if initial_final:
#                 text.append(split_phoneme_and_tone[0])
#             else:
#                 text.extend(split_phoneme_and_tone[0])
#     return text

# def parse_ipa_hat_tts(ipa: str):
#     text = []
#     delete_chars="\-\_"
#     # delete_chars="\+\-\|\_"
#     as_space=""
    
#     ipa_list = re.split(r"(?<![\d])(?=[\d])|(?<=[\d])(?![\d])", ipa)
#     for word in ipa_list:
#         if word.isdigit():
#             text.append(word)
#         else:
#             if len(as_space) > 0:
#                 word = re.sub(r"[{}]".format(as_space), " ", word)
#             if len(delete_chars) > 0:
#                 word = re.sub(r"[{}]".format(delete_chars), "", word)
                
#             word = word.replace("，", " ， ")
#             text.extend(word)

#     return text

if __name__ == "__main__":

    json_path_list = []
    json_lines = []
    for dataset in DATASET:
        json_path_list += glob.glob(f"{dataset}/fa*.json")

    print(json_path_list)
    for json_path in json_path_list:
        with open(json_path, "r") as f:
            json_lines += f.readlines()
    
    json_lines = f"[{','.join(json_lines)}]"
    items = json.loads(json_lines)

    ipa_set = set()
    for item in items:
        if "OOV" in item["text"]:
            continue

        # ipa = parse_ipa_hat_tts(item["ipa"])
        ipa = parse_ipa_hat_tts(item["ipa"])
        ipa_set.update(ipa)
    
    print(sorted(ipa_set))
        
    