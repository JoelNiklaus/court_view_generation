from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import IntervalStrategy, EarlyStoppingCallback

from scripts.util import get_batch_size

import os
import torch
from nltk.translate import meteor_score
import numpy as np
import argparse
import wandb
import json
import csv

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from evaluate import load

import logging

# implement timer
import time
import datetime

### Initialization
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", help="Want to finetune model?")
parser.add_argument("--model", help="Model name for finetune / evaluation (depends on finetune flag")
parser.add_argument("--train_size", help="Size of training set", type=int)
parser.add_argument("--eval_size", help="Size of evaluation set", type=int)
parser.add_argument("--test_size", help="Size of test set", type=int)
parser.add_argument("--seq_length", help="Sequence length for training, evaluation and generation", type=int)
parser.add_argument("--grad_acc_steps", help="Gradient accumulation steps for training", type=int)
parser.add_argument("--epochs", help="Number of training epochs", type=int)
parser.add_argument("--gm", help="GPU memory size for batch size", type=int)
args = parser.parse_args()

# print all args
logger.info(args)

batch_size = get_batch_size(args.model.split('/')[-1], args.gm, args.seq_length)

if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"device count: {torch.cuda.device_count()}")
    logger.info("Running on the GPU: " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    logger.info("Running on the CPU")

### Some Methods
def generate_text(model, tokenizer, input_text_encoded, attention_mask, max_length, num_return_sequences=1, temperature=1.0):
    input_tokens = torch.tensor(input_text_encoded).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    output_tokens = model.generate(
        input_tokens,
        max_length=max_length,
        attention_mask=attention_mask,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample_on_gen,
        temperature=temperature
    )
    output_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
    return output_texts[0], output_tokens

# Preprocess data
def preprocess_function(examples, max_length):
    input_texts = [f"facts: {f}" for f in examples["facts"]]
    target_texts = [f"considerations: {c}" for c in examples["considerations"]]

    input_encodings = [
        tokenizer.encode_plus(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length) for text
        in input_texts]
    target_encodings = [
        tokenizer.encode_plus(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length) for text
        in target_texts]

    inputs = {
        "input_ids": torch.cat([encoding["input_ids"] for encoding in input_encodings], dim=0),
        "attention_mask": torch.cat([encoding["attention_mask"] for encoding in input_encodings], dim=0),
    }
    targets = {
        "input_ids": torch.cat([encoding["input_ids"] for encoding in target_encodings], dim=0),
        "attention_mask": torch.cat([encoding["attention_mask"] for encoding in target_encodings], dim=0),
    }

    return {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
        "labels": targets["input_ids"].tolist(),
    }

def average_rouge_scores(rouge_scores_list):
    avg_scores = {
        'rouge-1': {'r': 0, 'p': 0, 'f': 0},
        'rouge-2': {'r': 0, 'p': 0, 'f': 0},
        'rouge-l': {'r': 0, 'p': 0, 'f': 0}
    }

    num_scores = len(rouge_scores_list)

    for scores in rouge_scores_list:
        for rouge_type in avg_scores:
            for metric in avg_scores[rouge_type]:
                avg_scores[rouge_type][metric] += scores[rouge_type][metric]

    for rouge_type in avg_scores:
        for metric in avg_scores[rouge_type]:
            avg_scores[rouge_type][metric] /= num_scores

    return avg_scores

def average_bert_score(bert_scores):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = len(bert_scores)

    for bert_score in bert_scores:
        total_precision += sum(bert_score['precision']) / len(bert_score['precision'])
        total_recall += sum(bert_score['recall']) / len(bert_score['recall'])
        total_f1 += sum(bert_score['f1']) / len(bert_score['f1'])

    return {
        'precision': total_precision / count,
        'recall': total_recall / count,
        'f1': total_f1 / count
    }

output_examples = []

def compute_scores(test_data, model, tokenizer, num_examples=100):
    scores = {'meteor': [], 'rouge': [], 'bleu': [], 'bert': []}
    rouge = Rouge()
    bertscore = load("bertscore")

    for idx, example in enumerate(test_data):
        input_text = example['input_ids']
        tokenized_target_text = example['labels']
        attention_mask = example['attention_mask']
        # measure time
        start_time_gen = time.time()
        predicted_text, tokenized_predicted_text = generate_text(model, tokenizer, input_text, attention_mask, max_length=args.seq_length, temperature=1)
        end_time_gen = time.time()

        target_text = tokenizer.decode(tokenized_target_text, skip_special_tokens=True)

        # convert tensor "tokenized_predicted_text" to list of tokens
        predicted_text_tokens = tokenizer.convert_ids_to_tokens(tokenized_predicted_text.tolist()[0])

        # convert ids to tokens
        target_text_tokens = tokenizer.convert_ids_to_tokens(tokenized_target_text)

        # Join tokens to form text strings
        tokenized_target_text = " ".join(target_text_tokens)
        tokenized_predicted_text = " ".join(predicted_text_tokens)

        """
        target_text_tokens: list of tokens
        predicted_text_tokens: list of tokens
        
        predicted_text: string (plain text)
        target_text: string (plain text)
        
        tokenized_target_text: string (tokens)
        tokenized_predicted_text: string (tokens)
        """

        # Calculate Meteor scores
        meteor = meteor_score.meteor_score([target_text_tokens], predicted_text_tokens)
        scores['meteor'].append(meteor)

        # Calculate Rouge scores
        rouge_scores = rouge.get_scores(predicted_text, target_text)[0]
        scores['rouge'].append(rouge_scores)

        # Calculate Bleu scores
        bleu = sentence_bleu([tokenized_target_text], tokenized_predicted_text, weights=(.25, .25, .25, .25))
        scores['bleu'].append(bleu)

        # Calculate BERTScore
        bert = bertscore.compute(predictions=[predicted_text], references=[target_text], model_type="bert-base-multilingual-cased",  lang=['de', 'fr', 'it'])
        scores['bert'].append(bert)

        output_examples.append({
            'target': target_text,
            'predicted': predicted_text,
            'meteor': meteor,
            'bert-f1': bert['f1'][0],
            'bleu': bleu,
            'rouge-1_f1': rouge_scores['rouge-1']['f'],
            'rouge-2_f1': rouge_scores['rouge-2']['f'],
            'rouge-l_f1': rouge_scores['rouge-l']['f'],
            'bert_full': bert,
            'rouge_full': rouge_scores,
        })

        # Print examples
        if idx < num_examples:
            print("\n", flush=True)
            print("#" * 180, flush=True)
            logger.info(f"Example {idx + 1} of {len(test_data)}")
            logger.info(f"Output: {predicted_text}")
            logger.info("-" * 100)
            logger.info(f"Label: {target_text}")
            logger.info("-" * 100)
            logger.info(f"METEOR Score: {meteor:.4f}")
            logger.info(f"ROUGE Score: {rouge_scores}")
            logger.info(f"BLEU Score: {bleu:.4f}")
            logger.info(f"BERTScore: {bert}")
            # log time
            logger.info(f"Time to generate: {end_time_gen - start_time_gen}")
            print("#" * 180, flush=True)
            print("\n", flush=True)

    return np.mean(scores['meteor']), average_rouge_scores(scores['rouge']), np.mean(scores['bleu']), average_bert_score(scores['bert'])

def load_model(model_name, tokenizer_name):
    if 'mt5' in model_name:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif 'mgpt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
        model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, tokenizer

def log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg):
    wandb.log({
        "METEOR_score/test": meteor_score_avg,
        "ROUGE_score/test/rouge-1/r": rouge_score_avg['rouge-1']['r'],
        "ROUGE_score/test/rouge-1/p": rouge_score_avg['rouge-1']['p'],
        "ROUGE_score/test/rouge-1/f": rouge_score_avg['rouge-1']['f'],
        "ROUGE_score/test/rouge-2/r": rouge_score_avg['rouge-2']['r'],
        "ROUGE_score/test/rouge-2/p": rouge_score_avg['rouge-2']['p'],
        "ROUGE_score/test/rouge-2/f": rouge_score_avg['rouge-2']['f'],
        "ROUGE_score/test/rouge-l/r": rouge_score_avg['rouge-l']['r'],
        "ROUGE_score/test/rouge-l/p": rouge_score_avg['rouge-l']['p'],
        "ROUGE_score/test/rouge-l/f": rouge_score_avg['rouge-l']['f'],
        "BLEU_score/test": bleu_score_avg,
        "BERT_score/test/precision": bert_score_avg['precision'],
        "BERT_score/test/recall": bert_score_avg['recall'],
        "BERT_score/test/f1": bert_score_avg['f1']
    })
    print()
    logger.info(f"Average METEOR score: {meteor_score_avg:.4f}")
    logger.info(f"Average ROUGE score: {rouge_score_avg}")
    logger.info(f"Average BLEU score: {bleu_score_avg:.4f}")
    logger.info(f"Average BERTScore: {bert_score_avg}")
    print()

def log_duration(start_time, end_time_train, end_time):
    # Calculate the time taken in seconds
    time_taken = end_time - start_time
    time_taken_train = end_time_train - start_time
    time_taken_eval = end_time - end_time_train
    # Print the time taken in hours, minutes, and seconds
    print()
    logger.info("Time taken: " + str(datetime.timedelta(seconds=time_taken)))
    logger.info("Time taken for training: " + str(datetime.timedelta(seconds=time_taken_train)))
    logger.info("Time taken for testing: " + str(datetime.timedelta(seconds=time_taken_eval)))
    print()
    # Convert the time taken to minutes
    time_taken_minutes = time_taken / 60
    time_taken_train_minutes = time_taken_train / 60
    time_taken_eval_minutes = time_taken_eval / 60
    # Log the time taken to wandb
    wandb.log({"time_taken": time_taken_minutes,
               "time_taken_train": time_taken_train_minutes,
               "time_taken_test": time_taken_eval_minutes})

def export_output(data):
    # Export to CSV
    with open(f"{output_dir}/output_{output_dir.split('/')[-1]}.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # Export to JSONL
    with open(f"{output_dir}/output_{output_dir.split('/')[-1]}.jsonl", mode='w') as jsonl_file:
        for row in data:
            json.dump(row, jsonl_file)
            jsonl_file.write('\n')

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# start timer
start_time = time.time()

finetune = args.finetune == "True"

model_name = args.model
tokenizer_name = args.model.split('_')[0]

model, tokenizer = load_model(model_name, tokenizer_name)

special_tokens_dict = {'pad_token': '<pad>', 'sep_token': '<sep>'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Set pad_token_id and eos_token_id for the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Load dataset
dataset = load_dataset("rcds/swiss_court_view_generation", "full")
train_dataset = dataset['train']
eval_dataset = dataset['validation']
test_dataset = dataset['test']

# Update args values with the full lengths of the dataset splits if the args values are -1
if args.train_size == -1:
    args.train_size = len(train_dataset)
if args.eval_size == -1:
    args.eval_size = len(eval_dataset)
if args.test_size == -1:
    args.test_size = len(test_dataset)

# Select subsets of the dataset based on the updated args values
train_dataset = train_dataset.select(range(args.train_size))
eval_dataset = eval_dataset.select(range(args.eval_size))
test_dataset = test_dataset.select(range(args.test_size))

# add train size, seq length to output dir
output_dir = f"output/{args.model.split('/')[-1]}_trainsize={args.train_size}_seqlen={args.seq_length}_batchsize={batch_size}_gaccsteps={args.grad_acc_steps}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
# set wandb run name
wandb.init(name=output_dir.split('/')[-1])
# log output dir to wandb
wandb.log({"output_dir": output_dir})

logger.info("Model name:" + model_name + " tokenizer: " + tokenizer_name + " finetune: " + str(finetune) + " output_dir: " + output_dir)
logger.info("Train dataset size: " + str(len(train_dataset)) + ", Eval dataset size: " + str(len(eval_dataset)) + ", Test dataset size: " + str(len(test_dataset)))

os.environ["WANDB_PROJECT"] = "court view generation"
os.environ["WANDB_RUN_GROUP"] = f"{model_name}, {len(train_dataset)}"

# generate by sampling or not
do_sample_on_gen = True
wandb.log({"do_sample_on_gen": do_sample_on_gen})

# log all args to wandb
wandb.config.update(args)


# Tokenize datasets
train_data = train_dataset.map(lambda x: preprocess_function(x, max_length=args.seq_length), batched=True)
eval_data = eval_dataset.map(lambda x: preprocess_function(x, max_length=args.seq_length), batched=True)
test_data = test_dataset.map(lambda x: preprocess_function(x, max_length=args.seq_length), batched=True)

model.resize_token_embeddings(len(tokenizer))
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=args.grad_acc_steps,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="logs",
    report_to="wandb",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

## Train model
if finetune:
    logger.info("Fine-tuning model..")
    # Fine-tune model
    trainer.train()
    # save model
    trainer.save_model(output_dir)
    logger.info("Model saved to " + output_dir)

end_time_train = time.time()

# Load model
if not finetune:
    model.to(device)

# Compute METEOR score
logger.info("testing model...")

model.eval() # set model to evaluation mode

# Evaluate model on test dataset
meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg = compute_scores(test_data, model, tokenizer)

# Print and log scores to wandb
log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg)

try:
    # save output examples to file
    export_output(output_examples)
except Exception as e:
    logger.info("Error exporting output examples: " + str(e))

# end timer
end_time = time.time()

# Log time taken to wandb and prints them (for train and eval)
log_duration(start_time, end_time_train, end_time)

print()
logger.info("Finished evaluation")
