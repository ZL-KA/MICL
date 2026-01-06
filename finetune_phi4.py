#Source: https://gist.github.com/seastar105/d1d8983b27611370528e3b194dcc5577#file-main-py
"""
finetune Phi-4-multimodal-instruct on an speech task
scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
from jiwer import cer, wer
import re
from whisper_normalizer.basic import BasicTextNormalizer
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
    EarlyStoppingCallback,
)
import random
import utils
from speechlm_icl_strategy import icl_prompt_audios_generation
from collections import defaultdict

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.
        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.
        """
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


class STTDataset_icl(Dataset):
    def __init__(self, processor, lang, num_sample_in_icl=1, rank=0, world_size=1, split='train'):

        self.ds = load_dataset(f"TBD", split='train')
        self.icl_pool_ds = self.ds.select(range(100))
        if split=='train':
            self.dataset = self.ds.select(range(100, len(self.ds)))
        else:
            self.dataset = load_dataset(f"TBD", split=split)
        self.processor = processor
        self.rank = rank
        self.world_size = world_size
        # self.instruction = "Transcribe the audio clip into text."
        self.training = True if "train" in split else False
        self.num_sample_in_icl=num_sample_in_icl
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Define prompt
        audios, transcripts = [], []
    
        icl_pool_ds = self.icl_pool_ds.shuffle()
        for idx_in_train in range(self.num_sample_in_icl):
            audios.append((icl_pool_ds[idx_in_train]['audio']['array'], icl_pool_ds[idx_in_train]['audio']['sampling_rate']))
            transcripts.append(icl_pool_ds[idx_in_train]['transcript'])
        prompt = "<|system|>You are a helpful and accurate AI model that transcribes audio clips into written text in one unknown language.<|end|>"
        for i, transcript in enumerate(transcripts):
            prompt += f"<|user|><|audio_{i+1}|>Transcribe the audio clip into text.<|end|>"
            prompt += f"<|assistant|>{transcript}<|end|>"
        prompt += f"<|user|><|audio_{len(transcripts)+1}|>Transcribe the audio clip into text.<|end|><|assistant|>"
        audios.append((data['audio']['array'], data['audio']['sampling_rate']))
        
        inputs = self.processor(text=prompt, audios=audios, return_tensors='pt')
        answer = f"{data['transcript']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }

class STTDataset_mlsuperb2(Dataset):
    def __init__(self, processor, dataset, pool_dataset, lang_indices, lang_id2idx, with_pairs, with_texts, without_target_audio, num_sample_in_icl, split, rank, world_size):
        self.dataset = dataset
        self.processor = processor
        self.pool_dataset = pool_dataset
        self.lang_indices = lang_indices
        self.lang_id2idx = lang_id2idx
        self.rank = rank
        self.world_size = world_size
        self.with_texts = with_texts
        self.with_pairs = with_pairs
        self.without_target_audio = without_target_audio
        self.training = True if "train" in split else False
        self.num_sample_in_icl=num_sample_in_icl
    def __len__(self):
        return len(self.dataset) 
    def __getitem__(self, idx): 
        data = self.dataset[idx]
        
        lang = data["language"]
        # Copy the candidate indices
        candidates = self.lang_indices[lang].copy()
        if self.training:
            # Remove the one with the same id
            remove_idx = self.lang_id2idx[lang][data["id"]]
            candidates.remove(remove_idx)   # O(n) but only one removal
        # Select the candidates from the pool dataset
        icl_pool_ds = self.pool_dataset.select(candidates) # 0.006s # Checked results with matching langs.
        
        if self.num_sample_in_icl<0: # Randomly select from 1 to -N or total sample number
            the_num_sample_in_icl=random.randint(1, min(-self.num_sample_in_icl, len(icl_pool_ds)))
        else:
            the_num_sample_in_icl=min(self.num_sample_in_icl, len(icl_pool_ds))
        
        if self.with_pairs:
            prompt_type=4
        elif self.with_texts:
            if self.without_target_audio:
                prompt_type=2
            else:
                prompt_type=3    
        else:
            raise ValueError("Please specify the with_texts or with_pairs option.")
        prompt, audios = icl_prompt_audios_generation(sample=data, task='transcription', prompt_type=prompt_type, num_sample=the_num_sample_in_icl, sample_selection_strategy='random', icl_pools=icl_pool_ds, sonar_results=None, sample_idx=None)
        if self.without_target_audio:
            inputs = self.processor(prompt, return_tensors="pt")
        else:
            inputs=self.processor(prompt, audios=audios, return_tensors="pt")
        
        answer = f"{data['transcript']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }
        
        
def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def collate_fn_icl(batch): 
    '''support multiple audio inputs in one sample'''
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        # audio_attention_mask_list is different to normal asr to support processing more than one audio in one sample
        assert inputs['input_audio_embeds'].shape[0]>1
        # audio_attention_mask_list.extend([inputs['input_audio_embeds'].new_full((seq.size(0),), True, dtype=torch.bool)
        #     for seq in inputs['input_audio_embeds']]) # This is wrong because the mask for different audio in one sample got the same masking
        for i in range(inputs['input_audio_embeds'].size(0)):
            x=inputs['input_audio_embeds'][i]
            try:
                # Compute which rows are all zeros
                row_is_zero = (x.abs().sum(dim=1) == 0)
                # Scan backward to find the first non-zero row (i.e., last valid frame) # Some 0 in the begining rows but not all 0 for that row
                last_valid_index = (row_is_zero == False).nonzero(as_tuple=False).max().item()
                # Build the mask (True for valid frames, False for padding)
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
                mask[:last_valid_index + 1] = True
            except:
            
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            audio_attention_mask_list.append(mask)
    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )

def collate_fn_notargetaudio(batch):
#     'input_ids': tensor([[200022,   3575,    553,    448,   8333,    540,   7524,    261,    620,
#            6439,    591,   5181,  10176,  40536,     13,   1608,    553,   1217,
#             261,  10297,    326,  16360,  20837,   2359,    484,   1643, 128055,
#           11065,  43018,   1511,   7582,   2201,    306,    484,   6439,    558]]), 'input_image_embeds': tensor([]), 'image_sizes': tensor([]), 'image_attention_mask': tensor([]), 'input_audio_embeds': tensor([]), 'audio_embed_sizes': tensor([]), 'audio_attention_mask': None, 'attention_mask': tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True]]), 'input_mode': tensor([0])}
    input_ids_list = []
    labels_list = []

    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])


    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)

    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    
    attention_mask = (input_ids != 0).long()

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_mode': 0,  # language mode
        }
    )


def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []

    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )



def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='ROOT/speechllm/ICL_speech/models/phi4/', help='Output directory')
    parser.add_argument('--lang', type=str, default='mboshi', help='dataset_name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--batch_size_per_gpu',type=int, default=16, help='Batch size per GPU (adjust this to fit in GPU memory)')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--do_icl_prompt_tuning', action='store_true')
    parser.add_argument('--do_icl_prompt_tuning_textonly', action='store_true')
    parser.add_argument('--do_icl_prompt_tuning_ppltype7', action='store_true')
    parser.add_argument('--num_sample_in_icl', type=int, default=1)
    parser.add_argument('--modules_to_ft', type=str, default='lora', choices=['lora', 'lora_projector', 'lora_projector_encoder', 'decoder'])
    parser.add_argument('--with_texts', action='store_true', help='Use texts in ICL prompt')
    parser.add_argument('--with_pairs', action='store_true', help='Use pairs in ICL prompt')
    parser.add_argument('--without_target_audio', action='store_true', help='use no target audio in ICL prompt')
    
    parser.add_argument('--saving_step', type=int, default=5000)

    args = parser.parse_args()
    print(args)
    utils.seed_everything(42)

    

    accelerator = Accelerator()
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluate_results=[]
    eval_interval=100
    trick_eval_size=100

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(args.model_name_or_path,trust_remote_code=True,)
        model = create_model(args.model_name_or_path, use_flash_attention=args.use_flash_attention)
    
    # Freeze all parameters
    print('Now freezing all parameters...')
    for param in model.parameters():
        param.requires_grad = False
    
    if args.modules_to_ft =='lora':
        model.set_lora_adapter('speech') # Only remove some params in LM
    elif args.modules_to_ft == 'lora_projector':
        model.set_lora_adapter('speech')
        for component in [model.model.embed_tokens_extend.audio_embed.audio_projection]:
            for param in component.parameters():
                param.requires_grad = True
    elif args.modules_to_ft == 'lora_projector_encoder':        
        model.set_lora_adapter('speech')
        for component in [model.model.embed_tokens_extend.audio_embed]: # This include audio_embed.encoder & audio_embed.audio_projection
            for param in component.parameters():
                param.requires_grad = True
    elif args.modules_to_ft == 'original_wrong_approach': # Before Aug 06, all exp did the following, which means all parameters are trainable except the vision lora
        for param in model.parameters():
            param.requires_grad = False  
        model.set_lora_adapter('speech')
    elif args.modules_to_ft == 'decoder':
        import pdb;pdb.set_trace()
        for component in [model.model.layers]:
            for param in component.parameters():
                param.requires_grad = True
        import pdb;pdb.set_trace()
    else:
        raise ValueError(f"Unknown modules_to_ft: {args.modules_to_ft}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M') # 461.37M
    print(f'Total parameters: {total_params / 1e6:.2f}M') # 5574.46M
    audio_embed_total_params = sum(p.numel() for p in model.model.embed_tokens_extend.audio_embed.parameters())
    audio_embed_trainable_params = sum(p.numel() for p in model.model.embed_tokens_extend.audio_embed.parameters() if p.requires_grad)
    print(f'Audio embed parameters: {audio_embed_total_params / 1e6:.2f}M') # 466.42M
    print(f'Audio embed trainable parameters: {audio_embed_trainable_params / 1e6:.2f}M') # 0.00M
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if args.lang=='ml_superb2' or args.lang in ['ml_superb2_supported8', 'ml_superb2_supported8_total16', 'ml_superb2_supported8_total32', 'ml_superb2_supported8_total64']:
        do_filter=True
        if do_filter:
            ds1= load_dataset('espnet/ml_superb_hf')
            print(ds1)
            ds1 = ds1.rename_columns({'text': 'transcript'})
            ds1 = ds1.map(utils.map_preprocess_mlsuperb_text)
            ds2=load_from_disk('ROOT/speechllm/ml_superb2_data/stats/mlsuperb2_stats')
            ds1 = ds1.remove_columns(['id','language'])
            merged = DatasetDict()
            for split in ds1.keys():
                merged[split] = concatenate_datasets([ds1[split], ds2[split]], axis=1)
            merged['train'] = merged['train'].filter(lambda x: x["duration"] >= 0.5 and len(x["transcript"]) > 0 and len(x["transcript"]) <= 200)
            dataset=merged
            print(dataset)
            
        else:
            dataset= load_dataset('espnet/ml_superb_hf')    
            dataset = dataset.rename_columns({'text': 'transcript'})
            dataset = dataset.map(utils.map_preprocess_mlsuperb_text)
        
        ds_split=dataset['train'].train_test_split(test_size=0.1, seed=42).shuffle() # Here shuffled before split
        train_dataset = ds_split['train']
        eval_dataset = ds_split['test']

        import pickle
        lang_indices_path="ROOT/speechllm/ml_superb2_data/training_mapping_raw/lang_indices.pkl"
        lang_id2idx_path="ROOT/speechllm/ml_superb2_data/training_mapping_raw/lang_id2idx.pkl"
        if do_filter:
            lang_indices_path=lang_indices_path.replace("lang_indices", "lang_indices_do_filter")
            lang_id2idx_path=lang_id2idx_path.replace("lang_id2idx", "lang_id2idx_do_filter")
        if args.lang in ['ml_superb2_supported8', 'ml_superb2_supported8_total16', 'ml_superb2_supported8_total32', 'ml_superb2_supported8_total64']:
            lang_id_to_select_map={
                'ml_superb2_supported8':0,
                'ml_superb2_supported8_total16':8,
                'ml_superb2_supported8_total32':24,
                'ml_superb2_supported8_total64':56,}
            supported8_lang_ids = ['eng','deu','fra','ita','spa','jpn','por','cmn']
            distinct_langs = list(set(train_dataset['language']))
            if lang_id_to_select_map[args.lang]==0:
                selected_lang = supported8_lang_ids
            else:
                filtered_langs = [lang for lang in distinct_langs if lang not in supported8_lang_ids]
                filtered_langs.sort()
                random.seed(42)
                random.shuffle(filtered_langs)
                selected_additional_langs = filtered_langs[:lang_id_to_select_map[args.lang]+1]
                selected_lang = supported8_lang_ids + selected_additional_langs
                print(selected_lang)
            with open(f"ROOT/speechllm/ml_superb2_data/training_mapping_raw/{args.lang}_langid.txt", "w", encoding="utf-8") as f:
                f.write(",".join(selected_lang))
            
            train_dataset = train_dataset.filter(lambda x: x['language'] in selected_lang)
            lang_indices_path=lang_indices_path.replace('.pkl', f'_{args.lang}.pkl')
            lang_id2idx_path=lang_id2idx_path.replace('.pkl', f'_{args.lang}_.pkl')

        
        print(f"Loading or creating lang_indices from {lang_indices_path}\n and lang_id2idx from {lang_id2idx_path}")
        if os.path.exists(lang_indices_path) and os.path.exists(lang_id2idx_path):
            with open(lang_indices_path, "rb") as f:
                lang_indices = pickle.load(f)
            with open(lang_id2idx_path, "rb") as f:
                lang_id2idx = pickle.load(f)
        else:
            print("lang_indices.pkl and lang_id2idx.pkl not found, creating new ones...")
            # Create lang_indices and lang_id2idx
            lang_indices = defaultdict(list)
            lang_id2idx = defaultdict(dict)
            for i, ex in enumerate(train_dataset): # This is slow but only run once with 2mins
                lang = ex["language"]
                lang_indices[lang].append(i)
                lang_id2idx[lang][ex["id"]] = i
            # Save with pickle
            with open(lang_indices_path, "wb") as f:
                pickle.dump(dict(lang_indices), f)
            with open(lang_id2idx_path, "wb") as f:
                pickle.dump(dict(lang_id2idx), f)
        
        import copy
        pool_dataset = copy.deepcopy(train_dataset) # To avoid the issue of map function changing the dataset
        train_dataset=STTDataset_mlsuperb2(processor=processor, dataset=train_dataset, pool_dataset=pool_dataset, lang_indices=lang_indices, lang_id2idx=lang_id2idx, with_pairs=args.with_pairs, with_texts=args.with_texts, without_target_audio=args.without_target_audio, num_sample_in_icl=args.num_sample_in_icl, split='train', rank=rank, world_size=world_size)
        # eval_dataset=STTDataset_mlsuperb2(processor=processor, dataset=eval_dataset, pool_dataset=pool_dataset, lang_indices=lang_indices, lang_id2idx=lang_id2idx, with_pairs=args.with_pairs, with_texts=args.with_texts, without_target_audio=args.without_target_audio, num_sample_in_icl=args.num_sample_in_icl, split='valid', rank=rank, world_size=world_size)
        if args.with_texts:
            if args.without_target_audio:
                data_collator_fn=collate_fn_notargetaudio
                # data_collator_fn=collate_fn
            else:
                data_collator_fn=collate_fn
        elif args.with_pairs:
            data_collator_fn=collate_fn_icl
        else:
            raise ValueError("Please specify the with_texts or with_pairs option.")

        
        

    
    else:

        if args.do_icl_prompt_tuning:
            eval_dataset = STTDataset_icl(processor, split='valid', num_sample_in_icl=args.num_sample_in_icl,lang=args.lang, rank=rank, world_size=world_size)
            train_dataset = STTDataset_icl(processor, split='train', num_sample_in_icl=args.num_sample_in_icl,lang=args.lang)
            data_collator_fn = collate_fn_icl # Support multiple audio inputs
        elif args.do_icl_prompt_tuning_textonly:
            eval_dataset = STTDataset_icl_textonly(processor, split='valid', num_sample_in_icl=args.num_sample_in_icl, lang=args.lang, rank=rank, world_size=world_size)
            train_dataset = STTDataset_icl_textonly(processor, split='train', num_sample_in_icl=args.num_sample_in_icl,lang=args.lang)
            data_collator_fn = collate_fn
        elif args.do_icl_prompt_tuning_ppltype7:
            eval_dataset = STTDataset_icl_ppltype7(processor, split='valid', num_sample_in_icl=args.num_sample_in_icl,lang=args.lang, rank=rank, world_size=world_size)
            train_dataset = STTDataset_icl_ppltype7(processor, split='train', num_sample_in_icl=args.num_sample_in_icl,lang=args.lang)
            data_collator_fn = collate_fn_icl
            aaa=collate_fn([train_dataset[1], train_dataset[2]])
        else:
            eval_dataset = STTDataset(processor, split='valid', lang=args.lang, rank=rank, world_size=world_size)
            train_dataset = STTDataset(processor, split='train', lang=args.lang)
            data_collator_fn = collate_fn
            # aaa=collate_fn([train_dataset[11], train_dataset[21]])
            # for key, value in aaa.items():
                # print(f"{key}: {value.shape}")
    

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)
    print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=500,
        logging_steps=100,
        save_steps=args.saving_step,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
        # per_device_eval_batch_size=1,
        # evaluation_strategy='steps',
        # eval_steps=10,
    )

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # score = evaluate(
    #     model,
    #     processor,
    #     eval_dataset,
    #     save_path=out_path / 'eval_before.json',
    #     disable_tqdm=not args.tqdm,
    #     eval_batch_size=args.batch_size_per_gpu,
    # )
    # if accelerator.is_main_process:
    #     print(f'CER Score before finetuning: {score}')

    from transformers import EarlyStoppingCallback

    # Use different collate function

    # # Trick to save evaluating time
    # if len(eval_dataset)>trick_eval_size:
    #     import random
    #     from torch.utils.data import Subset
    #     indices = list(range(len(eval_dataset)))
    #     random.seed(42)
    #     random.shuffle(indices)
    #     eval_dataset=Subset(eval_dataset, indices[:trick_eval_size])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator_fn,
        train_dataset=train_dataset,
        )




    trainer.train()

    trainer.save_model()

    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # Save evaluate_results
    import json
    evaluate_results_save_path=f'{training_args.output_dir}/evaluate_results_interval{eval_interval}.json'
    with open(evaluate_results_save_path, 'w') as f:
        json.dump(evaluate_results, f)
    
    # plot
    import os
    os.system(f'python ROOT/speechllm/ICL_speech/scripts/plot_eval_results.py {evaluate_results_save_path} {eval_interval}')

    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # # reload the model for inference
    # model = AutoModelForCausalLM.from_pretrained(
    #     training_args.output_dir,
    #     torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
    #     trust_remote_code=True,
    #     _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    # ).to('cuda')

    # score = evaluate(
    #     model,
    #     processor,
    #     eval_dataset,
    #     save_path=out_path / 'eval_after.json',
    #     disable_tqdm=not args.tqdm,
    #     eval_batch_size=args.batch_size_per_gpu,
    # )
    # if accelerator.is_main_process:
    #     print(f'CER Score after finetuning: {score}')


if __name__ == '__main__':
    main()