import math
import os
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCausalLM
import utils
from datasets import load_dataset
import json
import numpy as np
from tqdm import tqdm
import random
import argparse
import torch.nn.functional as F

# phi4 
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
system_prompt = '<|system|>'

# qwen3-omni
qwen3_omni_im_start_token = '<|im_start|>'
qwen3_omni_im_end_token = '<|im_end|>'
qwen3_omni_audio_start_token = '<|audio_start|>'
qwen3_omni_audio_end_token = '<|audio_end|>'
qwen3_omni_audio_pad_token = '<|audio_pad|>'

qwen3_omni_one_audio_tokens = f'{qwen3_omni_audio_start_token}{qwen3_omni_audio_pad_token}{qwen3_omni_audio_end_token}'
qwen3_omni_system_token = 'system'
qwen3_omni_user_token = 'user'
qwen3_omni_assistant_token = 'assistant'


system_prompt_textsonly = "You are an expert at learning the language from provided samples. "
system_prompt_with_target_audio = system_prompt_textsonly + "You are also a helpful and accurate AI model that transcribes audio clips into written text in that language."
separator = '\n' # Tested with <ss> and space, but found \n is better

saving_root='ROOTspeechllm/ICL_speech/exp_results'
abnormal_ppl_value=999999


def normalize_to_unit_range(values):
    """
    Normalize a list of floats to [0, 1] according to rules:
      - All positive: min -> 1, max -> 0 (inverted scale)
      - All negative: max -> 1, min -> 0
      - Mixed signs: raises ValueError
    """
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmin < 0 and vmax > 0:
        raise ValueError("Values contain both positive and negative numbers.")
    if vmin == vmax:
        return [1.0 for _ in values]  # everything identical
    if vmax <= 0:
        # all negatives: max → 1, min → 0
        return [(x - vmin) / (vmax - vmin) for x in values]
    else:
        # all positives: min → 1, max → 0 (invert)
        return [1 - (x - vmin) / (vmax - vmin) for x in values]

def load_sonar_results(sample_selection_strategy, dataset, lang_code=None):
    if sample_selection_strategy in ['sample_oracle', 'sample_best1', 'sample_audio', 'sample_audio_text_best1']:
        results = {}
        if dataset=='ml_superb2':
            file_paths={'hypo':f'ROOTspeechllm/ml_superb2_data/sonar_results/{lang_code}.json'} #TODO: This should be SONAR hypos
        else:
            file_paths = {
                "hypo": f'ROOTspeechllm/sonar_embedding_selection/{dataset}_sonar_cossimi_hypos_results.json',
                "oracle": f'ROOTspeechllm/sonar_embedding_selection/{dataset}_sonar_cossimi_text_oracle_results.json',
                "speech": f'ROOTspeechllm/sonar_embedding_selection/{dataset}_sonar_cossimi_speech_results.json'
            }
        for key, path in file_paths.items():
            with open(path, 'r') as f:
                if dataset=='ml_superb2':
                    # Now only one hypo
                    the_value= json.load(f)
                    results[key]=[[value] for value in the_value]
                else:
                    results[key]=json.load(f)
    else:
        results=None
    return results


def select_indices_for_icl(sample_selection_strategy, num_sample, total_len, sonar_results=None, sample_idx=None):
    if num_sample==0:
        return []
    if sonar_results is not None:
        hypo_results = sonar_results['hypo']
        try:
            oracle_results = sonar_results['oracle']
            speech_results = sonar_results['speech']
        except:
            oracle_results = None
            speech_results = None
    if sample_selection_strategy == 'firstn':
        indices=[i for i in range(num_sample)]
    elif sample_selection_strategy == 'random':
        indices = random.sample(range(total_len), num_sample)
    elif sample_selection_strategy=='corpus_best1':
        first_hypo = [item[0] for item in hypo_results]
        corpus_best1 = np.sum(np.array(first_hypo), axis=0).tolist()
        indices=np.argsort(corpus_best1)[-num_sample:][::-1].tolist()
    elif sample_selection_strategy =='sample_oracle':
        sonar_scores=oracle_results[sample_idx]
        indices= np.argsort(sonar_scores)[-num_sample:][::-1].tolist()
    elif sample_selection_strategy =='sample_best1':
        sample_hypo_result_best=hypo_results[sample_idx][0]
        indices=np.argsort(sample_hypo_result_best)[-num_sample:][::-1].tolist()
    elif sample_selection_strategy == 'sample_audio':
        sonar_scores=speech_results[sample_idx]
        indices= np.argsort(sonar_scores)[-num_sample:][::-1].tolist()
    elif sample_selection_strategy =='sample_audio_text_best1':
        sample_hypo_result_best=hypo_results[sample_idx][0]
        sonar_scores=speech_results[sample_idx]
        total_scores = [a+b for a,b in zip(sample_hypo_result_best,sonar_scores)]
        indices= np.argsort(total_scores)[-num_sample:][::-1].tolist()
    else:
        import pdb;pdb.set_trace()
    return indices

def icl_prompt_audios_generation(sample, task, prompt_type, num_sample, sample_selection_strategy, icl_pools, sonar_results=None, sample_idx=None, what_model_prompt=None):
    """
    Prompt design:
    1. System + texts + task
    2. Texts
    3. System + texts + Task + target audio
    4. System + Pairs + Task + target audio
    Notes for design: the phi4-mm paper says: We notice that the model can learn to recognize in the 
    target language perfectly without providing language information, while Qwen2-audio and Gemini-2.0-Flash 
    require the language information in the prompt to obtain the optimal ASR performance
    """
    assert what_model_prompt in ['phi4','qwen3-omni']
    # Select icl samples
    indices = select_indices_for_icl(sample_selection_strategy=sample_selection_strategy, num_sample=num_sample, total_len=len(icl_pools), sonar_results=sonar_results, sample_idx=sample_idx)
    selected_samples = icl_pools.select(indices)

    # Generate prompt and audios
    prompt = ''

    # Define task prompt
    if task=='transcription' or task=='ppl_calculation' or task=='ppl_hypo_selection' or task=='ppl_hypo_calculation' or task=='visualize_attention':
        task_prompt = f'Transcribe the audio clip into text and output only the transcription.'
    elif task== 'hypo_selection':
        hypos=sample['10_best']
        the_hypos = f"{separator}".join(hypos)
        task_prompt = f'Select the best transcription from the following hypotheses for the audio clip and provide only the selection:{separator}{the_hypos}{separator}'
    else:
        raise ValueError(f'Unknown task: {task}')

    # Prompt for phi4
    if what_model_prompt=='phi4':
        if prompt_type==1:
            prompt += f'{system_prompt}{system_prompt_textsonly}'
            prompt += f'{user_prompt}Here are sample sentences:{separator}'
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            prompt += f'The target sentence is:{separator}'
            audios=[]
        elif prompt_type==2:
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            audios=[]
        elif prompt_type==3:
            prompt += f'{system_prompt}{system_prompt_with_target_audio}'
            prompt += f'{user_prompt}Here are sample sentences:{separator}'
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            prompt += f'<|audio_1|>' + task_prompt + f'{prompt_suffix}{assistant_prompt}'
            audios = [(sample['audio']['array'], sample['audio']['sampling_rate'])] # target audio
        elif prompt_type==4:
            prompt += f'{system_prompt}{system_prompt_with_target_audio}'
            prompt += f'{user_prompt}Here are sample pairs of audio and text:{separator}'
            audios = []
            for i in range (len(selected_samples)):
                if i == 999999: # For testing replacing one sample with gold transcript
                    # prompt += f'<|audio_{i+1}|>Transcription: <PLACEHOLDER>' # Target text
                    prompt += f'<|audio_{i+1}|>Transcription: {selected_samples[i]["transcript"]}{separator}'

                    audios.append((sample['audio']['array'], sample['audio']['sampling_rate'])) # Target audio
                    # audios.append((selected_samples[i]['audio']['array'], selected_samples[i]['audio']['sampling_rate']))
                else:
                    prompt += f'<|audio_{i+1}|>Transcription: {selected_samples[i]["transcript"]}{separator}'
                    audios.append((selected_samples[i]['audio']['array'], selected_samples[i]['audio']['sampling_rate']))
            prompt += f'<|audio_{len(selected_samples)+1}|>' + task_prompt + f'{prompt_suffix}{assistant_prompt}'
            audios.append((sample['audio']['array'], sample['audio']['sampling_rate'])) # target audio
        else:
            import pdb;pdb.set_trace()
    # Prompt for qwen3-omni
    elif what_model_prompt=='qwen3-omni':
        if prompt_type==1:
            prompt += f'{qwen3_omni_im_start_token}{system_prompt}{separator}{system_prompt_textsonly}{qwen3_omni_im_end_token}{separator}'
            prompt += f'{qwen3_omni_im_start_token}{qwen3_omni_user_token}{separator}Here are sample sentences:{separator}'
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            prompt += f'The target sentence is:{qwen3_omni_im_end_token}{separator}{qwen3_omni_im_start_token}{qwen3_omni_assistant_token}'
            audios=[]
        elif prompt_type==2:
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            audios=[] 
        elif prompt_type==3:
            prompt += f'{qwen3_omni_im_start_token}{system_prompt}{separator}{system_prompt_with_target_audio}{qwen3_omni_im_end_token}{separator}'
            prompt += f'{qwen3_omni_im_start_token}{qwen3_omni_user_token}{separator}Here are sample sentences:{separator}'
            prompt += f'{separator}'.join(selected_samples['transcript']) + f'{separator}'
            prompt += f'{qwen3_omni_one_audio_tokens}' + task_prompt + f'{qwen3_omni_im_end_token}{separator}{qwen3_omni_im_start_token}{qwen3_omni_assistant_token}'
            audios = [sample['audio']['array']]
            pass
        elif prompt_type==4:
            prompt += f'{qwen3_omni_im_start_token}{system_prompt}{separator}{system_prompt_with_target_audio}{qwen3_omni_im_end_token}{separator}'
            prompt += f'{qwen3_omni_im_start_token}{qwen3_omni_user_token}{separator}Here are sample pairs of audio and text:{separator}'
            audios = []
            for i in range (len(selected_samples)):
                prompt += f'{qwen3_omni_one_audio_tokens}Transcription: {selected_samples[i]["transcript"]}{separator}'
                audios.append(selected_samples[i]['audio']['array'])
            prompt += f'{qwen3_omni_one_audio_tokens}' + task_prompt + f'{qwen3_omni_im_end_token}{separator}{qwen3_omni_im_start_token}{qwen3_omni_assistant_token}'
            audios.append(sample['audio']['array'])
        else:
            import pdb;pdb.set_trace()
    else:
        import pdb;pdb.set_trace()
    return prompt, audios
def asr_prompt_audios_generation(sample, what_model_prompt=None):
    if what_model_prompt=='phi4':
        prompt = f'{user_prompt}<|audio_1|>Transcribe the audio clip into text.{prompt_suffix}{assistant_prompt}'
        audios=[(sample['audio']['array'], sample['audio']['sampling_rate'])]
    elif what_model_prompt=='qwen3-omni':
        prompt = f'{qwen3_omni_im_start_token}{qwen3_omni_user_token}{separator}{qwen3_omni_one_audio_tokens}Please provide the transcription for the audio clip.{qwen3_omni_im_end_token}{separator}{qwen3_omni_im_start_token}{qwen3_omni_assistant_token}'
        audios=[sample['audio']['array']]
    return prompt, audios


def collect_attention_dfs(prompt_inputs, outputs, folder_path=None):
    """
    Return a list of df for each layer, where each df has columns [prompt, pairs, target audio, task, generated tokens] and rows are generated tokens
    """
    # folder_path = 'ROOTspeechllm/ICL_speech/exp_results/visual_attention_test_10samples'
    # os.makedirs(folder_path, exist_ok=True)
    all_ids = prompt_inputs.input_ids[0].cpu().tolist()
    
    dfs = []
    # Generated tokens & Items
    for n_layer in range(len(outputs.attentions)):
        attn_layer = outputs.attentions[n_layer][0]
        
        # # Option to plot signgle head of the layer
        # for n_head in range(attn_layer.shape[0]):
        #     save_path = f'{folder_path}/layer{n_layer}_head{n_head}.png'
        #     df = utils.collect_scores_as_df(attn_layer[n_head], all_ids)
        #     plot_attn(df, save_path)
        
        
        # Mean of heads
        attn_mean = attn_layer.mean(dim=0) # average heads
        df = utils.collect_scores_as_df(attn_mean, all_ids)
        dfs.append(df)
        
        # # Option to plot all mean heads of the layer
        # save_path = f'{folder_path}/layer{n_layer}_headmean.png'        
        # plot_attn(df, save_path)
        
    return dfs

    
def call_processor(text, audios, what_model_prompt=None):
    if what_model_prompt=='phi4':
        # import pdb; pdb.set_trace() # Check if it works after changing the way of calling function
        if audios== []:
            inputs = processor(text, return_tensors="pt")
        else:
            inputs = processor(text, audios=audios, return_tensors="pt")
    elif what_model_prompt=='qwen3-omni':
        if audios== []:
            inputs = processor(text=text, return_tensors="pt")
        else:
            inputs = processor(text=text, audio=audios, return_tensors="pt")
    return inputs

def calculate_ppl(model, processor, prompt_text, sample_text, audios,what_model_prompt, do_attn_visual=False):
    
    if '<PLACEHOLDER>' in prompt_text:
        # To test replacing one sample with gold transcript
        prompt_text = prompt_text.replace('<PLACEHOLDER>', sample_text)
    
    prompt_inputs = call_processor(prompt_text, audios, what_model_prompt).to(device).to(model.dtype)
    sample_inputs = call_processor(sample_text, [], what_model_prompt).to(device).to(model.dtype)
    valid_logits_start=prompt_inputs.input_ids.shape[1]

    cat_prompt_inputs_ids=torch.cat((prompt_inputs.input_ids, sample_inputs.input_ids), dim=1)
    cat_attention_mask=torch.cat((prompt_inputs.attention_mask, sample_inputs.attention_mask), dim=1)
    prompt_inputs['input_ids'] = cat_prompt_inputs_ids
    prompt_inputs['attention_mask']=cat_attention_mask

    if what_model_prompt=='qwen3-omni':
        model=model.thinker # Qwen3-omni use thinker as the name of LM decoder
    
    with torch.no_grad():
        if do_attn_visual:
            outputs = model(**prompt_inputs, output_attentions=True)
            attn_df = collect_attention_dfs(prompt_inputs, outputs)
        else:
            outputs = model(**prompt_inputs)
        
        logits = outputs.logits
        logits = logits[:, :-1, :]  # Remove the last token logits
        labels = prompt_inputs.input_ids[:, 1:]  # Remove the first token labels

        logits = logits[:, valid_logits_start:, :]
        labels = labels[:, valid_logits_start:]
        # Flatten for cross_entropy
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),labels.reshape(-1),ignore_index=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else -100, reduction='mean')
        if math.isnan(loss.item()):
            print(f'Found nan ppl')
            ppl=abnormal_ppl_value
        else:
            ppl= math.exp(loss.item())

        if do_attn_visual:
            return attn_df
        else:
            return ppl



def load_hypos(dataset_name, test_dataset, lang):
    if dataset_name in ['mboshi','khinalug','kichwa']:
        hypothesis_path = f'ROOTspeechllm/datasets/{dataset_name}_wav2vec2_10best.json'
        with open(hypothesis_path, "r", encoding="utf-8") as f:
            hypo_ds = json.load(f)
        hypo_asr_scores = [item['logit_score'] for item in hypo_ds]
        hypo_text = [item['text'] for item in hypo_ds]
        test_dataset = test_dataset.add_column('10_best', hypo_text)
        test_dataset = test_dataset.add_column('10_best_asr_scores', hypo_asr_scores)
    elif dataset_name == 'ml_superb2':
        lang_id=lang.replace('ml_superb2/','')
        hypothesis_path = f'ROOTspeechllm/ml_superb2_data/whisper_hypos_with_langid/{lang_id}_sample.json'
        with open(hypothesis_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        test_dataset = test_dataset.add_column('10_best', loaded)
        test_dataset = test_dataset.map(utils.map_preprocess_mlsuperb_hypo) # process the hypos here

    else:
        import pdb;pdb.set_trace()
    return test_dataset

def do_icl(task, prompt_type, num_samples, sample_selection_strategy, results_saving_prefix, test_dataset, icl_pools, lang, sonar_results=None, what_model_prompt=None):
    # Iterate over the number of samples to do ICL
    ppls_all=[]
    if 'ml_superb2' in lang:
        saving_folder_path = f'{saving_root}/{lang}_'
    else:
        saving_folder_path = f'{saving_root}/{lang}/'
        os.makedirs(saving_folder_path, exist_ok=True)
    if task=='asr' or task=='asr_ppl': # Ugly forced setting
        num_samples=[9999]
    # Load hypos if needed
    if task=='ppl_hypo_selection' or task=='hypo_selection' or task=='ppl_hypo_calculation':
        test_dataset=load_hypos(dataset_name=args.dataset, test_dataset=test_dataset, lang=lang)


    # Do ICL
    # Iterate over the number of samples
    for num_sample in num_samples:
        # Iterate over the test_dataset
        ppl_hypo_calculation_results = []
        transcripts = []
        transcripts_combine_acoustic_llm=[]
        ppls = []
        attn_dfs = []
        for sample_idx, sample in enumerate(tqdm(test_dataset)):
            # Generate inputs from prompt and audios
            if task=='asr' or task=='asr_ppl':
                prompt, audios = asr_prompt_audios_generation(sample, what_model_prompt=what_model_prompt)
            else:
                prompt, audios = icl_prompt_audios_generation(sample, task, prompt_type, num_sample, sample_selection_strategy, icl_pools, sonar_results=sonar_results, sample_idx=sample_idx, what_model_prompt=what_model_prompt)
            # Do task

            if task=='ppl_calculation' or task=='asr_ppl':
                # Calculate ppl
                ppl = calculate_ppl(model, processor, prompt, sample['transcript'], audios, what_model_prompt)
                if ppl==abnormal_ppl_value:
                    continue
                else:
                    ppls.append(ppl)
                     
            elif task=='visualize_attention':
                attn_df = calculate_ppl(model, processor, prompt, sample['transcript'], audios, what_model_prompt, do_attn_visual=True)
                attn_dfs.append(attn_df)

            elif task=='ppl_hypo_selection' or task=='ppl_hypo_calculation':
                
                hypo_ppls=[]
                for hypo in sample['10_best']:
                    ppl = calculate_ppl(model, processor, prompt, hypo, audios, what_model_prompt)
                    hypo_ppls.append(ppl)
                if task=='ppl_hypo_calculation':
                    ppl_hypo_calculation_results.append({"text":sample['10_best'],'llm_ppls':hypo_ppls})
                if task=='ppl_hypo_selection':
                    # Only LLM score
                    best_hypo_index = int(np.argmin(hypo_ppls))
                    transcripts.append(sample['10_best'][best_hypo_index])
                    # Combine LLM and acoustic score
                    normalize_ppls = normalize_to_unit_range(hypo_ppls)
                    try:
                        normalize_acoustic = normalize_to_unit_range(sample['10_best_asr_scores'])
                    except:
                        normalize_acoustic = [0]*len(normalize_ppls)
                    combined_scores = [a + l for a, l in zip(normalize_acoustic, normalize_ppls)]
                    best_hypo_index = combined_scores.index(max(combined_scores))
                    transcripts_combine_acoustic_llm.append(sample['10_best'][best_hypo_index])
            else:
                # Generation
                inputs = call_processor(prompt, audios, what_model_prompt).to(device).to(model.dtype)
                
                generate_ids = model.generate(**inputs, no_repeat_ngram_size=3, max_new_tokens=200)
                if isinstance(generate_ids, tuple):
                    generate_ids = generate_ids[0] # Qwen3-omni returns a tuple of thinker and takler results
                
                generated_text = processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                if '\n' in generated_text:
                    # print(f'Found new_line_symbol in generated text; removed now')
                    generated_text = generated_text.replace('\n','')
                transcripts.append(generated_text)
        # Save results for the current num_sample
        if not args.no_saving:
            if transcripts!=[]:
                if task in ['asr', 'asr_ppl']:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}.txt'
                else:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_prompt{prompt_type}_{num_sample}shot_{sample_selection_strategy}samples.txt'
                print('Saving results to ', results_saving_path)
                with open(results_saving_path, 'w') as f:
                    for item in transcripts:
                        f.write(f"{item}\n")
            if transcripts_combine_acoustic_llm!=[]:
                if task in ['asr', 'asr_ppl']:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_llm_asr_combined_selection.txt'
                else:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_prompt{prompt_type}_{num_sample}shot_{sample_selection_strategy}samples_llm_asr_combined_selection.txt'
                print('Saving results to ', results_saving_path)
                with open(results_saving_path, 'w') as f:
                    for item in transcripts_combine_acoustic_llm:
                        f.write(f"{item}\n")
                
            if ppl_hypo_calculation_results!=[]:
                if task in ['asr', 'asr_ppl']:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}.json'
                else:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_prompt{prompt_type}_{num_sample}shot_{sample_selection_strategy}samples_hypo_ppls.json'
                print('Saving results to ', results_saving_path)
                with open(results_saving_path, 'w') as f:
                    json.dump(ppl_hypo_calculation_results, f, indent=4)
            

            # Save the overall ppls in every num_sample loop to avoid loss due to crashes
            if ppls!=[]:
                try:
                    ppls_all.append(int(sum(ppls)/len(ppls)))
                except:
                    import pdb;pdb.set_trace()
                    ppls_all.append(-1) # Nan for empty ppls
                if task in ['asr', 'asr_ppl']:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}.json'
                else:
                    results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_prompt{prompt_type}_{sample_selection_strategy}samples.json'
                print('Saving results to ', results_saving_path)
                with open(results_saving_path, 'w') as f:
                    data = {'num_samples': num_samples, 'avg_ppl': ppls_all}
                    json.dump(data, f, indent=4)
            
            if attn_dfs!=[]:
                import pickle
                results_saving_path = f'{saving_folder_path}{results_saving_prefix}_{task}_prompt{prompt_type}_{num_sample}shot_{sample_selection_strategy}samples.pkl'
                print('Saving results to ', results_saving_path)
                with open(results_saving_path, "wb") as f:
                    pickle.dump(attn_dfs, f)

            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='khinalug', choices=['ml_superb2', 'mboshi', 'khinalug', 'kichwa', 'fleurs'])
    parser.add_argument('--lang_code_list', nargs='+', type=str, help='list of language code of ml-superb2 to run exp', default=[])
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    # Model
    parser.add_argument('--model_name', type=str)
    # Processing
    parser.add_argument('--sample_selection_strategy', type=str, choices=['random', 'corpus_best1', 'sample_oracle', 'sample_best1', 'sample_audio', 'sample_audio_text_best1'])
    parser.add_argument('--prompt_type', type=int)
    parser.add_argument('--task', type=str, choices=['transcription', 'hypo_selection', 'ppl_calculation', 'ppl_hypo_selection','asr', 'asr_ppl', 'ppl_hypo_calculation', 'visualize_attention'])
    parser.add_argument('--num_samples',  nargs='+', type=int, default=[9999]) # default for asr relating tasks
    # Saving
    parser.add_argument('--results_saving_prefix', type=str, default='')
    parser.add_argument('--no_saving', action='store_true', help='If set, no saving')


    args = parser.parse_args()
    print(args)
    utils.seed_everything()
    

    # Load the processor and model    
    if 'qwen2-audio' in args.model_name.lower():
        import pdb;pdb.set_trace()
        from transformers import Qwen2AudioForConditionalGeneration
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    elif 'qwen3-omni' in args.model_name.lower():
        MODEL_PATH = args.model_name
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if args.task=='visualize_attention': # for visual attention, use eager mode to get attn weights which is not supported in flash attention
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype='auto', _attn_implementation='eager')
        else:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype='auto', _attn_implementation='flash_attention_2')
        # model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map="auto", _attn_implementation='flash_attention_2')
        what_model_prompt = 'qwen3-omni'
        model.disable_talker() # Remove talker to save memory as we do not need it here
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.eos_token_id
    elif 'qwen2.5-omni' in args.model_name.lower():
        MODEL_PATH=args.model_name
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map="auto", _attn_implementation='flash_attention_2')
        what_model_prompt = 'qwen3-omni'
        model.disable_talker()

    elif 'phi' in args.model_name.lower():
        what_model_prompt = 'phi4'
        processor = AutoProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)
        
        if args.task=='visualize_attention': # for visual attention, use eager mode to get attn weights which is not supported in flash attention
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype='auto', _attn_implementation='eager')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype='auto', _attn_implementation='flash_attention_2')
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    # Do ICL
    
    if args.dataset=='ml_superb2':
        dataset= load_dataset('espnet/ml_superb_hf')
        dataset = dataset.rename_columns({'text': 'transcript'})
        dataset_test = dataset['dev']
        dataset_train=dataset['train']
        
        import pdb;pdb.set_trace()
        if args.lang_code_list==['all']:
            args.lang_code_list = sorted(list(set(dataset_test['language'])))
        assert len(args.lang_code_list)>0
        for lang_code in args.lang_code_list:
            print(f'Processing language: {lang_code}')
            the_dataset_train=dataset_train.filter(lambda example: example['language'] == lang_code)
            the_dataset_test= dataset_test.filter(lambda example: example['language'] == lang_code)
            print('the_dataset_train: ', the_dataset_train)
            print('the_dataset_test: ', the_dataset_test)
            # preprocess transcript
            the_dataset_train = the_dataset_train.map(utils.map_preprocess_mlsuperb_text)
            the_dataset_test = the_dataset_test.map(utils.map_preprocess_mlsuperb_text)
            # the_dataset_test = the_dataset_test.select(range(1))

            # trim the list of number of samples to do ICL
            the_samples = [num for num in args.num_samples if num<=len(the_dataset_train)]
            print('The num_samples to do ICL: ', the_samples)

            sonar_results=load_sonar_results(args.sample_selection_strategy,args.dataset, lang_code=lang_code)
            # Do ICL
            do_icl(
                task=args.task, 
                prompt_type=args.prompt_type,
                num_samples=the_samples, 
                sample_selection_strategy=args.sample_selection_strategy,
                results_saving_prefix=args.results_saving_prefix,
                test_dataset=the_dataset_test,
                icl_pools=the_dataset_train,
                lang='ml_superb2/'+lang_code,
                sonar_results=sonar_results,
                what_model_prompt=what_model_prompt
                )
    elif args.dataset=='fleurs':
        import pdb;pdb.set_trace()
    else:
        assert args.dataset in ['mboshi', 'khinalug', 'kichwa']
        dataset = args.dataset
        dataset = load_dataset(f'TBD')
        if args.split=='valid':
            the_dataset_test = dataset['valid']
            results_saving_prefix = args.results_saving_prefix + '_validsplit'
        else:
            the_dataset_test = dataset['test']
            results_saving_prefix = args.results_saving_prefix
        the_dataset_train=dataset['train']
        sonar_results=load_sonar_results(args.sample_selection_strategy, args.dataset)
        do_icl(
                task=args.task, 
                prompt_type=args.prompt_type,
                num_samples=args.num_samples, 
                sample_selection_strategy=args.sample_selection_strategy,
                results_saving_prefix=results_saving_prefix, 
                test_dataset=the_dataset_test,
                icl_pools=the_dataset_train,
                lang=args.dataset,
                sonar_results=sonar_results,
                what_model_prompt=what_model_prompt
                )