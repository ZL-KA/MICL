from datasets import Dataset, load_dataset, Audio
from evaluate import load
wer=load("wer")
cer=load('cer')

import sys
import utils

lang=sys.argv[1]
pred_path = sys.argv[2]

print_details= False
# print_details= True

# import pdb;pdb.set_trace()

import os
import re
def normalize_standard(text):
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:\[\]')
    text = re.subn(english_filter, '', text)[0].lower()
    return text

if lang=='ml_superb2':
    # print(sys.argv[3])
    # import pdb;pdb.set_trace()
    ground_truth_file=f'ROOT/speechllm/ICL_speech/exp_results/ground_truth/{lang}/{sys.argv[3]}.txt'
    if not os.path.exists(ground_truth_file):
        import pdb;pdb.set_trace()

        # dataset= load_dataset('espnet/ml_superb_hf')
        # dataset = dataset.rename_columns({'text': 'transcript'})
        # dataset_test = dataset['dev']
        # the_dataset_test= dataset_test.filter(lambda example: example['language'] == sys.argv[3], load_from_cache_file=False)
        # the_dataset_test = the_dataset_test.map(utils.map_preprocess_mlsuperb_text, fn_kwargs={'lang_code': sys.argv[3]}, load_from_cache_file=False)
        # ground_truth=[normalize_standard(text) for text in the_dataset_test['transcript']]
        # with open(ground_truth_file, 'w') as f:
        #     for item in ground_truth:
        #         f.write(f"{item}\n")
else:
    ground_truth_file=f'TBD/speechllm/ICL_speech/exp_results/ground_truth/{lang}.txt'
    if not os.path.exists(ground_truth_file):
        import huggingface_hub
        dataset = load_dataset(f'TBD')
        with open(ground_truth_file, 'w') as f:
            for item in dataset['test']['transcript']:
                f.write(f"{item}\n")



with open(pred_path, 'r') as f:
    predictions = f.read().splitlines()
with open(ground_truth_file, 'r') as f:
    references = f.read().splitlines()


if not len(predictions)==len(references):
    print(f'pred: {len(predictions)}, ref: {len(references)}')



predictions=[normalize_standard(text) for text in predictions]

cer_result=round(cer.compute(references=references, predictions=predictions)*100, 1)
wer_result=round(wer.compute(references=references, predictions=predictions)*100, 1)

# print('\n')
# print(f'CER: {cer_result}')
print(f'{os.path.basename(sys.argv[2])}: CER: {cer_result}; WER: {wer_result}')
if print_details:
    print(f'lowest three cer: {sorted([(round(cer.compute(references=[r], predictions=[p]) * 100, 1), r, p) for r, p in zip(references, predictions)], key=lambda x: x[0])[:3]}')
    print(f'highest three cer: {sorted([(round(cer.compute(references=[r], predictions=[p]) * 100, 1), r, p) for r, p in zip(references, predictions)], key=lambda x: x[0], reverse=True)[:3]}')
    print(f'lowest three wer: {sorted([(round(wer.compute(references=[r], predictions=[p]) * 100, 1), r, p) for r, p in zip(references, predictions)], key=lambda x: x[0])[:3]}')
    print(f'highest three wer: {sorted([(round(wer.compute(references=[r], predictions=[p]) * 100, 1), r, p) for r, p in zip(references, predictions)], key=lambda x: x[0], reverse=True)[:3]}')