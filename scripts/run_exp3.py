from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from transformers import IntervalStrategy, EarlyStoppingCallback

import os
import torch
from nltk.translate import meteor_score
import numpy as np
import argparse
import wandb

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from evaluate import load

# implement timer
import time
import datetime

wandb.init()

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", help="Want to finetune model?")
parser.add_argument("--model", help="Model name for finetune / evaluation (depends on finetune flag")
parser.add_argument("--train_size", help="Size of training set (-1 for full)", type=int)
parser.add_argument("--eval_size", help="Size of evaluation set (-1 for full)", type=int)
parser.add_argument("--seq_length", help="Sequence length for training, evaluation and generation", type=int)
parser.add_argument("--batch_size", help="Batch size for training and evaluation", type=int)
parser.add_argument("--grad_acc_steps", help="Gradient accumulation steps for training", type=int)
parser.add_argument("--epochs", help="Number of training epochs", type=int)
args = parser.parse_args()

# print all args
print(args, flush=True)

# log all args to wandb
wandb.config.update(args)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.device_count())
    print("Running on the GPU", torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")
    print("Running on the CPU", flush=True)


def generate_text(model, tokenizer, input_text, max_length, num_return_sequences=1, temperature=1.0):
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_tokens = model.generate(
        input_tokens,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature,
        eos_token_id=None
    )
    print(len(output_tokens))
    output_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
    return output_texts[0]

# Preprocess data
def preprocess_function(examples, max_length):
    input_texts = [f"facts: {f}" for f in examples["facts"]]
    target_texts = [f"considerations: {c}" for c in examples["considerations"]]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
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


def compute_scores(eval_data, model, tokenizer, num_examples=10):
    scores = {'meteor': [], 'rouge': [], 'bleu': [], 'bert': []}
    rouge = Rouge()
    bertscore = load("bertscore")

    for idx, example in enumerate(eval_data):
        input_text = f"facts: {example['facts']}"
        target_text = f"considerations: {example['considerations']}"
        predicted_text = generate_text(model, tokenizer, input_text, max_length=args.seq_length, temperature=1)

        # Tokenize predicted_text and target_text
        tokenized_predicted_text = tokenizer.tokenize(predicted_text)
        tokenized_target_text = tokenizer.tokenize(target_text)[:args.seq_length]

        meteor = meteor_score.meteor_score([tokenized_target_text], tokenized_predicted_text)
        scores['meteor'].append(meteor)


        # Calculate Rouge scores
        rouge_scores = rouge.get_scores(predicted_text, target_text)[0]
        scores['rouge'].append(rouge_scores)

        # Calculate Bleu scores
        bleu = sentence_bleu(tokenized_target_text, tokenized_predicted_text, weights=(1, 0, 0, 0))
        scores['bleu'].append(bleu)

        # Calculate BERTScore
        bert = bertscore.compute(predictions=[predicted_text], references=[target_text], lang="de")
        scores['bert'].append(bert)

        # Print examples
        if idx < num_examples:
            # detokenize tokenized_target_text
            detokenized_target_text = tokenizer.convert_tokens_to_string(tokenized_target_text)
            # get first args.seq_length tokens of input_text but convert to string
            input_text_cutoff = tokenizer.convert_tokens_to_string(tokenizer.tokenize(input_text)[:args.seq_length])

            print(f"Example {idx + 1}:")
            print(f"Input: {input_text_cutoff}")
            print(f"Output: {predicted_text}")
            print(f"Label: {detokenized_target_text}")
            print(f"METEOR Score: {meteor:.4f}")
            print(f"ROUGE Score: {rouge_scores}")
            print(f"BLEU Score: {bleu:.4f}")
            print(f"BERTScore: {bert}")
            print("-" * 180)

    return np.mean(scores['meteor']), average_rouge_scores(scores['rouge']), np.mean(scores['bleu']), average_bert_score(scores['bert'])


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

model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# add train size, seq length to output dir
output_dir = f"output/{model_name.split('/')[-1]}_{args.train_size}_{args.seq_length}_{args.batch_size}_{args.grad_acc_steps}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"

special_tokens_dict = {'pad_token': '<pad>', 'sep_token': '<sep>'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

print("Model name:", model_name, "tokenizer:", tokenizer, "finetune:", finetune, output_dir, flush=True)


# Load dataset
dataset = load_dataset("rcds/swiss_court_view_generation", "full")
train_dataset, eval_dataset = dataset['train'].select(range(args.train_size)), dataset['validation'].select(range(args.eval_size))
print("Train dataset size:", len(train_dataset), "Eval dataset size:", len(eval_dataset), flush=True)

os.environ["WANDB_PROJECT"] = "court view generation"
os.environ["WANDB_RUN_GROUP"] = f"{model_name}, {len(train_dataset)}"


# Add special tokens to the tokenizer
special_tokens_dict = {'pad_token': '<pad>', 'sep_token': '<sep>'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Tokenize datasets
train_data = train_dataset.map(lambda x: preprocess_function(x, max_length=args.seq_length), batched=True)
eval_data = eval_dataset.map(lambda x: preprocess_function(x, max_length=args.seq_length), batched=True)

model.resize_token_embeddings(len(tokenizer))
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc_steps,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
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
    print("Fine-tuning model..", flush=True)
    # Fine-tune model
    trainer.train()
    # save model
    trainer.save_model(output_dir)
    print("Model saved to", output_dir, flush=True)

end_time_train = time.time()

# Load model
if not finetune:
    model.to(device)

# Compute METEOR score
print("evaluating model...", flush=True)

model.eval() # set model to evaluation mode

# Evaluate model
meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg = compute_scores(eval_dataset, model, tokenizer)

# log METEOR score to wandb
wandb.log({"METEOR_score/eval": meteor_score_avg})
print(f"Average METEOR score: {meteor_score_avg:.4f}")
wandb.log({"ROUGE_score/eval": rouge_score_avg})
print(f"Average ROUGE score: {rouge_score_avg}")
wandb.log({"BLEU_score/eval": bleu_score_avg})
print(f"Average BLEU score: {bleu_score_avg:.4f}")
wandb.log({"BERT_score/eval": bert_score_avg})
print(f"Average BERTScore: {bert_score_avg}")

# end timer
end_time = time.time()
# print time taken in hours, minutes, seconds
print("Time taken:", str(datetime.timedelta(seconds=end_time-start_time)))
# print time for training in hours, minutes, seconds
print("Time taken for training:", str(datetime.timedelta(seconds=end_time_train-start_time)))
# print time taken for evaluation in hours, minutes, seconds
print("Time taken for evaluation:", str(datetime.timedelta(seconds=end_time-end_time_train)))


