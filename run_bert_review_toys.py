from __future__ import absolute_import

import argparse
import gc
import logging
import os
import random
from io import open
from itertools import cycle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)  # ERROR
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, weight=None, reward=None, title_len=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.weight = weight
        self.reward = reward
        self.title_len = title_len


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label, weight, reward, title_len

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        self.weight = weight
        self.reward = reward
        self.title_len = title_len


def fun(c):
    c = c.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
    res = []
    for i in c.split(','):
        res.append(float(i))
    return res


def read_examples(input_file, is_training):
    df = pd.read_csv(input_file)
    examples = []
    if 'reward' not in df.columns: df['reward'] = '[1.0,1.0,1.0]'
    def_weight = [1.0, 1.0, 1.0]
    for val in df[['id', 'new_content', 'title', 'label', 'weight', 'reward']].values:
        val[3] = int(val[3]) - 1
        action = list(map(float, val[5][1:-1].split(',')))

        reward = 1.0  # 0-th episode
        if is_training:
            factor = 0.8
            reward = reward * factor + action[val[3]] * (1 - factor)  # # 1-th episode

        tmp = str(val[1]).split('|')
        title_len = len(val[2])
        examples.append(InputExample(guid=val[0], text_a=tmp, text_b=str(val[2]), label=val[3], weight=def_weight, reward=reward, title_len=title_len))
    return examples


def cut(text, k):
    # cut for k segments.
    max_len = int(len(text) / k)
    texts = []
    for i in range(k):
        if i == k - 1:
            texts.append(text[max_len * i:])
        else:
            texts.append(text[i * max_len:(i + 1) * max_len])
    return texts


def convert_examples_to_features(examples, tokenizer, max_seq_length, split_num,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    global label, weight, reward, title_len
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = []
        for i in range(split_num):
            context_tokens.append(tokenizer.tokenize(example.text_a[i]))
        ending_tokens = tokenizer.tokenize(example.text_b)
        choices_features = []
        for i in range(split_num):
            context_tokens_choice = context_tokens[i]
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens_choice + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            weight = torch.tensor(example.weight)
            reward = torch.tensor(example.reward)
            title_len = torch.tensor(example.title_len)
            if example_index < 1 and is_training:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))
                logger.info("weight: {}".format(weight))
                logger.info("reward: {}".format(reward))
                logger.info("title_len: {}".format(title_len))

        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label,
                weight=weight,
                reward=reward,
                title_len=title_len
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def f1(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs, labels=[0, 1, 2], average='macro')


def acc(out, labels):
    outputs = np.argmax(out, axis=1)
    return (outputs == labels).mean()


def precision(out, labels):
    outputs = np.argmax(out, axis=1)
    return precision_score(labels, outputs, labels=[0, 1, 2], average='macro')


def recall(out, labels):
    outputs = np.argmax(out, axis=1)
    return recall_score(labels, outputs, labels=[0, 1, 2], average='macro')


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--meta_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_acc", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--lstm_hidden_size", default=300, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="")

    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=3, type=int,
                        help="text split")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    try:
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir_acc)
    except:
        pass

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=5)
    config.output_hidden_states = True
    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, args, config=config)

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.do_train:

        # Prepare data loader

        train_examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training=True)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, args.split_num, True)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        for i, f in enumerate(train_features):
            if i == 0:
                all_weight = f.weight.unsqueeze(0)
            else:
                all_weight = torch.cat([all_weight, f.weight.unsqueeze(0)])
        all_reward = torch.tensor([f.reward for f in train_features], dtype=torch.float)
        all_title_len = torch.tensor([f.title_len for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_weight, all_reward, all_title_len)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.train_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_acc = 0
        real_best_acc = 0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)

        stop_gate = 0
        for step in bar:

            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, weight_ids, reward_ids, title_lens = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, weight_ids=weight_ids,
                         reward_ids=reward_ids, title_lens=title_lens)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))

            if args.do_eval and (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:

                for file in ['dev.csv', 'test.csv']:
                    inference_labels = []
                    gold_labels = []
                    inference_logits = []
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training=False)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, args.split_num, False)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                    for i, f in enumerate(eval_features):
                        if i == 0:
                            all_weight = f.weight.unsqueeze(0)
                        else:
                            all_weight = torch.cat([all_weight, f.weight.unsqueeze(0)])
                    all_reward = torch.tensor([f.reward for f in eval_features], dtype=torch.float)
                    all_title_len = torch.tensor([f.title_len for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_weight, all_reward, all_title_len)

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_f1, eval_acc, eval_recall, eval_precision = 0, 0, 0, 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids, weight_ids, reward_ids, title_lens in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        weight_ids = weight_ids.to(device)
                        reward_ids = reward_ids.to(device)
                        title_lens = title_lens.to(device)

                        with torch.no_grad():
                            tmp_eval_loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                                                  weight_ids=weight_ids, reward_ids=reward_ids, title_lens=title_lens)
                            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, weight_ids=weight_ids,
                                           reward_ids=reward_ids, title_lens=title_lens)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_labels = np.concatenate(gold_labels, 0)
                    inference_logits = np.concatenate(inference_logits, 0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_f1 = f1(inference_logits, gold_labels)
                    eval_acc = acc(inference_logits, gold_labels)
                    eval_recall = recall(inference_logits, gold_labels)
                    eval_precision = precision(inference_logits, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_F1': eval_f1,
                              'eval_acc': eval_acc,
                              'eval_recall': eval_recall,
                              'eval_precision': eval_precision,
                              'global_step': global_step,
                              'loss': train_loss}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if abs(eval_f1 - best_acc) < 0.05 and eval_f1 < best_acc and 'dev' in file:
                        stop_gate += 1
                    if eval_f1 > best_acc and 'dev' in file:
                        print("=" * 80)
                        print("\033[35mBest F1", eval_f1, '\033[0m')
                        print("Accuracy", eval_acc)
                        print("Saving Model......")
                        best_acc = eval_f1
                        stop_gate = 0
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, str(eval_f1) + "_pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("=" * 80)
                    else:
                        print("=" * 80)
                    if eval_acc > real_best_acc and 'dev' in file:
                        print("=" * 80)
                        print("\033[35mBest Accuracy", eval_acc, '\033[0m')
                        print("Fl", eval_f1)
                        print("Saving Model......")
                        real_best_acc = eval_acc
                        stop_gate = 0
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file_acc = os.path.join(args.output_dir_acc, str(eval_acc) + "_pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file_acc)
                        print("=" * 80)
                    else:
                        print("=" * 80)
            if stop_gate >= 100 and global_step > num_train_optimization_steps * 0.6:
                print("Early Stopping. Bye~")
                break
    if args.do_test:
        del model
        gc.collect()
        args.do_train = False
        model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), args, config=config)

        model.to(device)
        if args.local_rank != -1:
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        for file, flag in [('dev.csv', 'dev'), ('test.csv', 'test')]:
            inference_labels = []
            gold_labels = []
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training=False)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, args.split_num, False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

            for i, f in enumerate(eval_features):
                if i == 0:
                    all_weight = f.weight.unsqueeze(0)
                else:
                    all_weight = torch.cat([all_weight, f.weight.unsqueeze(0)])
            all_reward = torch.tensor([f.reward for f in eval_features], dtype=torch.float)
            all_title_len = torch.tensor([f.title_len for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_weight, all_reward, all_title_len)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            for input_ids, input_mask, segment_ids, label_ids, weight_ids, reward_ids, title_lens in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                title_lens = title_lens.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, title_lens=title_lens).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(logits)
                gold_labels.append(label_ids)
            gold_labels = np.concatenate(gold_labels, 0)
            logits = np.concatenate(inference_labels, 0)
            print(flag, f1(logits, gold_labels), acc(logits, gold_labels))
            if flag == 'test':
                df = pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0'] = logits[:, 0]
                df['label_1'] = logits[:, 1]
                df['label_2'] = logits[:, 2]
                df[['id', 'label_0', 'label_1', 'label_2']].to_csv(os.path.join(args.output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
