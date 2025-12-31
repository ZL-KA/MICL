import random
import numpy as np
import os
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import re # Python regex
import string # Python string manipulation library
import unicodedata # unicode punctuation detection
def remove_punctuation(sentence):
    '''https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview#top'''
    new_sentence = ""
    for char in sentence:
        # all unicode punctuation is of type P
        if unicodedata.category(char).startswith('P'):
            continue
        else:
            new_sentence = f"{new_sentence}{char}"
    return new_sentence



def ml_superb_text_normalization(text, lang_code):    
    text = text.lower()
    text = text.replace(f'[{lang_code}]', '').strip()
    text = remove_punctuation(text)
    # remove space for Chinese/Japanese/Thai
    if lang_code in ['tha', 'jpn', 'cmn']:
        text = re.sub(r"\s", "", text)
    return text

def map_preprocess_mlsuperb_hypo(example):
    lang_code = example['language']
    hypos=example['10_best']
    normalized_hypos = [ml_superb_text_normalization(hypo, lang_code) for hypo in hypos]
    example['10_best'] = normalized_hypos
    return example

def map_preprocess_mlsuperb_text(example):
    lang_code=example['language']
    example['transcript'] = ml_superb_text_normalization(example['transcript'], lang_code)
    return example








def load_fleurs_dataset(fleurs_langs_to_test=['Mandarin Chinese', 'English', 'Spanish', 'French', 'German', 'Italian', 'Japanese', 'Portuguese'], num_sample_in_icl_pool=300):
    from collections import OrderedDict
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    _FLEURS_LANG_TO_ID = OrderedDict([("Afrikaans", "af"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"), ("Assamese", "as"), ("Asturian", "ast"), ("Azerbaijani", "az"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"), ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"), ("Cebuano", "ceb"), ("Mandarin Chinese", "cmn_hans"), ("Cantonese Chinese", "yue_hant"), ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"), ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"), ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"), ("Fula", "ff"), ("Galician", "gl"), ("Ganda", "lg"), ("Georgian", "ka"), ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"), ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"), ("Japanese", "ja"), ("Javanese", "jv"), ("Kabuverdianu", "kea"), ("Kamba", "kam"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"), ("Kyrgyz", "ky"), ("Lao", "lo"), ("Latvian", "lv"), ("Lingala", "ln"), ("Lithuanian", "lt"), ("Luo", "luo"), ("Luxembourgish", "lb"), ("Macedonian", "mk"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"), ("Northern-Sotho", "nso"), ("Norwegian", "nb"), ("Nyanja", "ny"), ("Occitan", "oc"), ("Oriya", "or"), ("Oromo", "om"), ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"), ("Russian", "ru"), ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"), ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Sorani-Kurdish", "ckb"), ("Spanish", "es"), ("Swahili", "sw"), ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"), ("Ukrainian", "uk"), ("Umbundu", "umb"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"), ("Welsh", "cy"), ("Wolof", "wo"), ("Xhosa", "xh"), ("Yoruba", "yo"), ("Zulu", "zu")])
    _FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}
    _FLEURS_LANG = sorted(["af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in", "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr", "ckb_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oc_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru", "bg_bg", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za"])
    _FLEURS_LONG_TO_LANG = {_FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k for k in _FLEURS_LANG}
    _FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}
    collect_ds = {'train': [], 'valid': [], 'test': []}
    icl_pools = {lang: [] for lang in fleurs_langs_to_test}
    for lang in fleurs_langs_to_test:    
        the_ds = load_dataset("google/fleurs", _FLEURS_LONG_TO_LANG[lang]) # Already normalized
        the_ds = the_ds.remove_columns([name for name in the_ds['train'].column_names if name not in ['audio', 'transcription', 'language']])
        the_ds = the_ds.rename_column('transcription', 'transcript')
        # Select the first num_sample_in_icl_pool samples for ICL pool
        icl_pools[lang] = the_ds['train'].select(range(num_sample_in_icl_pool))
        # collect the rest 
        collect_ds['train'].append(the_ds['train'].select(range(num_sample_in_icl_pool, len(the_ds['train']))))
        collect_ds['valid'].append(the_ds['validation'])
        collect_ds['test'].append(the_ds['test'])
    
    # Combine the datasets
    combined_dict = {split: concatenate_datasets(dataset_list) for split, dataset_list in collect_ds.items()}
    final_dataset = DatasetDict(combined_dict)
    return final_dataset, icl_pools







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns






def collect_scores_as_df(attn_value, all_ids):
    """
    Collect scores into a dataframe for visualization.
    Rows: generated tokens
    Columns: prompt, samples (audio w/wo text), audioT, task, generated tokens
    """

    # Find pairs positions
    TOKEN_AUDIO = 200011
    TOKEN_END = 200020 #<|end|>
    TOKEN_ASSISTANT = 200019 #<|assistant|>
    TOKEN_BEGIN_SAMPLES = 734
    TOKEN_END_SAMPLE = 198
    
    # Find prompt boundary    
    indices_end_sample = [i for i, n in enumerate(all_ids) if n == TOKEN_END_SAMPLE] #'\n' for 'Transcription: xxxxxxx\n'
    indices_beginsamples = [i for i, n in enumerate(all_ids) if n == TOKEN_BEGIN_SAMPLES] #':/n' for 'here are the samples:\n'
    assert len(indices_beginsamples)==1  # Should be exact one due to prompt design
    assert indices_beginsamples[0] < indices_end_sample[0]  # The first end sample should be after begin samples
    prompt_end = indices_beginsamples[0]
    prompt_range = (0, prompt_end)

    # Find sample ranges
    samples = []
    last_end = prompt_end
    for end_idx in indices_end_sample:
        # A sample starts right after previous end (or begin_samples)
        start_idx = last_end + 1
        if TOKEN_AUDIO in all_ids[start_idx:end_idx]:  # Having audio in sample
            sub_ids = all_ids[start_idx:end_idx]
            # Find last occurrence
            last_audio_idx = len(sub_ids) - 1 - sub_ids[::-1].index(TOKEN_AUDIO)
            samples.append((start_idx, start_idx+last_audio_idx, start_idx+last_audio_idx+1, end_idx))  # (text_start, text_end, audio_start, audio_end)
        else: # No audio in sample
            samples.append((start_idx, end_idx))
        last_end = end_idx
    
    # Find target audio range
    start_target_audio = indices_end_sample[-1] + 1
    sub_ids = all_ids[start_target_audio: all_ids.index(TOKEN_END)]
    end_target_audio = start_target_audio + len(sub_ids) - 1 - sub_ids[::-1].index(TOKEN_AUDIO) +1
    
    target_audio_range = (start_target_audio, end_target_audio)

    # Find instruction range
    instruction_range = (end_target_audio, all_ids.index(TOKEN_END))

    # Find generated tokens range
    search_start_idx = all_ids.index(TOKEN_ASSISTANT) + 1
    search_indices = list(range(search_start_idx, attn_value.shape[-1]))

    
    # Collect scores and build heatmap
    rows, columns = [], []
    # row_names = [processor.decode(id) for id in search_indices]
    row_names = [f'token{i+1}' for i in range(len(search_indices))]
    # Columns
    columns += ['Prompt']
    for idx, sample in enumerate(samples):
        if len(sample)==2: # no audio
            columns +=[f'Text{idx+1}']
        else:
            columns +=[f'Audio{idx+1}']
            columns +=[f'Text{idx+1}']
    columns +=['AudioT']
    columns +=['Task']
    columns += row_names
    # Rows
    for idx in search_indices:
        row = []
        query_attn = attn_value[idx]
        row.append(query_attn[prompt_range[0]:prompt_range[1]].sum().item())
        # Samples
        for idx, sample in enumerate(samples):
            if len(sample)==2: # no audio
                row.append(query_attn[sample[0]:sample[1]].sum().item())
            else:
                row.append(query_attn[sample[0]:sample[1]].sum().item())
                row.append(query_attn[sample[2]:sample[3]].sum().item())
        # Target audio
        row.append(query_attn[target_audio_range[0]:target_audio_range[1]].sum().item())
        # Instruct
        row.append(query_attn[instruction_range[0]:instruction_range[1]].sum().item())
        row += [query_attn[item].item() for item in search_indices]
        # Predicted tokens
        rows.append(row)
    df = pd.DataFrame(rows, index=row_names, columns=columns)
    return df


def plot_attn(df, save_path):
    # === Plot heatmap ===
    plt.figure(figsize=(14, max(4, len(df) * 0.5)))
    sns.heatmap(df, cmap="viridis", annot=False, cbar=True)
    plt.title("Self attn heatmap")
    plt.xlabel("Keys")
    plt.ylabel("Query")
    plt.tight_layout()
    # import pdb;pdb.set_trace()
    plt.savefig(save_path, dpi=300)
    plt.close()


mlsuperb2_to_whisper = {
    "grn": "not_available",
    "yor": "yoruba",
    "mon": "mongolian",
    "azz": "not_available",
    "bos": "bosnian",
    "lga": "not_available",
    "kin": "not_available",
    "msa": "malay",
    "ven": "not_available",
    "ukr": "ukrainian",
    "fil": "tagalog",
    "glg": "galician",
    "cym": "welsh",
    "org_jpn": "japanese",
    "tgk": "tajik",
    "mrj": "not_available",
    "tso": "not_available",
    "tsn": "not_available",
    "tok": "not_available",
    "zul": "not_available",
    "epo": "not_available",
    "afr": "afrikaans",
    "kmr": "not_available",
    "oci": "occitan",
    "ast": "not_available",
    "hye": "armenian",
    "ita": "italian",
    "orm": "not_available",
    "ssw": "not_available",
    "lin": "lingala",
    "som": "somali",
    "dan": "danish",
    "fin": "finnish",
    "nep": "nepali",
    "umb": "not_available",
    "sun": "sundanese",
    "aze": "azerbaijani",
    "luo": "not_available",
    "hrv": "croatian",
    "isl": "icelandic",
    "mlt": "maltese",
    "ory": "not_available",
    "amh": "amharic",
    "fas": "persian",
    "kab": "not_available",
    "snd": "sindhi",
    "tur": "turkish",
    "swe": "swedish",
    "kaz": "kazakh",
    "nob": "norwegian",
    "uig": "not_available",
    "jpn": "japanese",
    "mya": "myanmar",
    "sot": "not_available",
    "nan": "chinese",
    "kor": "korean",
    "slk": "slovak",
    "por": "portuguese",
    "cat": "catalan",
    "yue": "cantonese",
    "wol": "not_available",
    "kan": "kannada",
    "est": "estonian",
    "pus": "pashto",
    "ibo": "not_available",
    "cnh": "not_available",
    "tel": "telugu",
    "hsb": "not_available",
    "kir": "not_available",
    "guj": "gujarati",
    "eng": "english",
    "xty": "not_available",
    "swa": "swahili",
    "mkd": "macedonian",
    "sna": "shona",
    "gle": "not_available",
    "ckb": "not_available",
    "kea": "not_available",
    "tha": "thai",
    "spa": "spanish",
    "khm": "khmer",
    "chv": "not_available",
    "ind": "indonesian",
    "ron": "romanian",
    "hun": "hungarian",
    "jav": "javanese",
    "bul": "bulgarian",
    "ara": "arabic",
    "ina": "not_available",
    "mri": "maori",
    "xho": "not_available",
    "skr": "not_available",
    "kam": "not_available",
    "slv": "slovenian",
    "tat": "tatar",
    "nbl": "not_available",
    "ori": "not_available",
    "ces": "czech",
    "sah": "not_available",
    "lao": "lao",
    "vie": "vietnamese",
    "frr": "not_available",
    "uzb": "uzbek",
    "bel": "belarusian",
    "mhr": "not_available",
    "nno": "nynorsk",
    "ben": "bengali",
    "asm": "assamese",
    "bak": "bashkir",
    "ell": "greek",
    "ceb": "not_available",
    "heb": "hebrew",
    "pan": "punjabi",
    "nor": "norwegian",
    "myv": "not_available",
    "ful": "not_available",
    "div": "not_available",
    "srp": "serbian",
    "tam": "tamil",
    "eus": "basque",
    "nld": "dutch",
    "lit": "lithuanian",
    "urd": "urdu",
    "tos": "not_available",
    "mal": "malayalam",
    "lug": "not_available",
    "pol": "polish",
    "bas": "not_available",
    "bre": "breton",
    "fra": "french",
    "rus": "russian",
    "cmn": "chinese",
    "ltz": "luxembourgish",
    "lav": "latvian",
    "sin": "sinhala",
    "deu": "german",
    "hin": "hindi",
    "abk": "not_available",
    "kat": "georgian",
    "nso": "not_available",
    "nya": "not_available",
    "mar": "marathi",
    "hau": "hausa",
}


mlsuperb2_train_dict = {'eng': 6049, 'nld': 4445, 'deu': 3988, 'fra': 2601, 'ita': 2586, 'rus': 2532, 'ben': 2503, 'spa': 2490, 'por': 2176, 'gle': 2135, 'pol': 2082, 'xho': 1834, 'swa': 1769, 'swe': 1746, 'sot': 1669, 'tam': 1627, 'mal': 1589, 'slk': 1576, 'slv': 1557, 'ces': 1545, 'glg': 1538, 'afr': 1432, 'nan': 1406, 'ukr': 1383, 'ron': 1372, 'fin': 1354, 'mar': 1339, 'kat': 1327, 'cat': 1321, 'hun': 1296, 'nep': 1290, 'lav': 1244, 'est': 1212, 'kir': 1149, 'ara': 1140, 'eus': 1125, 'ell': 1115, 'urd': 1108, 'nso': 1108, 'bre': 1104, 'wol': 1079, 'ind': 1078, 'yor': 1071, 'ckb': 1068, 'tha': 1038, 'abk': 1025, 'ori': 1017, 'yue': 1003, 'hau': 1000, 'hin': 977, 'mlt': 974, 'cmn': 946, 'bul': 941, 'cym': 936, 'fas': 924, 'uzb': 923, 'guj': 903, 'hye': 902, 'ven': 889, 'cnh': 886, 'kab': 882, 'asm': 881, 'bel': 867, 'jav': 854, 'mon': 841, 'tat': 835, 'bas': 827, 'sin': 815, 'amh': 810, 'skr': 802, 'grn': 773, 'kan': 771, 'azz': 766, 'ssw': 762, 'ina': 756, 'nbl': 744, 'chv': 738, 'kmr': 723, 'mrj': 707, 'org_jpn': 694, 'mhr': 686, 'bak': 667, 'kin': 656, 'div': 650, 'myv': 635, 'hrv': 595, 'lga': 583, 'xty': 580, 'uig': 571, 'sah': 560, 'hsb': 544, 'jpn': 509, 'heb': 345, 'mkd': 332, 'ast': 330, 'isl': 316, 'pan': 291, 'ltz': 284, 'tel': 278, 'aze': 273, 'kor': 269, 'snd': 263, 'pus': 255, 'msa': 249, 'tgk': 249, 'mya': 230, 'nya': 224, 'sna': 222, 'lao': 217, 'orm': 211, 'khm': 206, 'som': 193, 'fil': 185, 'kam': 184, 'oci': 180, 'ibo': 136, 'mri': 120, 'lug': 117, 'lin': 111, 'dan': 5, 'epo': 5, 'frr': 5, 'kaz': 5, 'tok': 5, 'vie': 5, 'bos': 5, 'ceb': 5, 'ful': 5, 'kea': 5, 'luo': 5, 'umb': 5, 'zul': 5, 'sun': 5, 'tsn': 5, 'tos': 5, 'tso': 5, 'lit': 3, 'srp': 3, 'tur': 3}

mlsuperb2_dev_dict = {'eng': 1042, 'nld': 787, 'deu': 645, 'rus': 484, 'fra': 449, 'ben': 417, 'ita': 416, 'spa': 405, 'por': 373, 'gle': 360, 'pol': 359, 'dan': 330, 'mal': 308, 'swa': 307, 'xho': 306, 'swe': 302, 'ces': 290, 'tsn': 283, 'tam': 279, 'sot': 277, 'glg': 262, 'slv': 247, 'ron': 243, 'slk': 242, 'afr': 240, 'srp': 237, 'nep': 233, 'ukr': 232, 'mar': 230, 'lav': 227, 'fin': 226, 'kat': 226, 'cat': 223, 'hun': 215, 'nso': 213, 'ind': 203, 'nan': 203, 'vie': 203, 'urd': 197, 'ell': 196, 'ara': 195, 'eus': 188, 'kir': 183, 'bre': 179, 'ckb': 179, 'yue': 179, 'est': 178, 'tur': 175, 'asm': 172, 'tha': 172, 'yor': 172, 'bul': 171, 'tos': 171, 'wol': 170, 'mlt': 169, 'cnh': 167, 'jpn': 166, 'lit': 166, 'tso': 166, 'hin': 165, 'tok': 162, 'cmn': 161, 'uzb': 161, 'zul': 159, 'mon': 155, 'guj': 153, 'hye': 150, 'bel': 149, 'amh': 148, 'kab': 148, 'bas': 145, 'fas': 144, 'hau': 143, 'cym': 139, 'sin': 137, 'abk': 136, 'grn': 135, 'jav': 135, 'kaz': 133, 'kan': 132, 'ven': 132, 'kmr': 130, 'tat': 130, 'nbl': 128, 'nno': 126, 'skr': 124, 'ina': 122, 'chv': 120, 'ory': 116, 'nor': 114, 'mrj': 114, 'azz': 113, 'bak': 112, 'epo': 112, 'kin': 111, 'mhr': 110, 'hrv': 110, 'sun': 110, 'div': 109, 'lga': 108, 'frr': 107, 'org_jpn': 106, 'ssw': 103, 'xty': 99, 'hsb': 97, 'myv': 97, 'sah': 91, 'uig': 91, 'heb': 70, 'ast': 67, 'msa': 60, 'lao': 57, 'pan': 54, 'ltz': 51, 'mkd': 51, 'tel': 51, 'kor': 49, 'pus': 49, 'aze': 48, 'bos': 47, 'snd': 47, 'som': 46, 'ori': 45, 'ful': 44, 'kea': 44, 'nob': 43, 'luo': 42, 'sna': 41, 'tgk': 39, 'mya': 38, 'isl': 36, 'oci': 36, 'nya': 34, 'khm': 33, 'lug': 33, 'ceb': 32, 'kam': 32, 'ibo': 22, 'mri': 22, 'fil': 18, 'lin': 18, 'orm': 16, 'umb': 10}