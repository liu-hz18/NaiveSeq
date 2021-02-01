import os
import time
import argparse
import torch

from torch.nn.utils import clip_grad_norm_
from apex import amp
from tqdm import tqdm
from .model import Transformer
from .optim import Adam, RAdam, SGD
from .optim.lr_scheduler import InverseSquareRootScheduler
from .data import Dictionary, Seq2SeqDataset, Seq2SeqDataLoader
from .utils import load_config_to_args, save_args_to_config

optimizer_map = {
    'adam': Adam,
    'radam': RAdam,
    'sgd': SGD,
}


class AutoRegressiveSeq2Seq(object):

    def __init__(self, args, model, optimizer, scheduler, train_dataloader, eval_dataloader, src_dict, tgt_dict):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        self.logging_dir = args.logging_dir + '_' + timestamp
        os.makedirs(self.logging_dir, exist_ok=False)
        save_args_to_config(self.args, os.path.join(self.logging_dir, 'config.json'))

    @classmethod
    def build_parser(cls):
        parser = argparse.ArgumentParser(description="Auto Regressive Seq2Seq Task Config")
        parser.add_argument("--data-dir", type=str, help="Dataset Path of MT")
        parser.add_argument("--train-file", type=str,
                            help="train dataset raw file")
        parser.add_argument("--eval-file", type=str,
                            help="eval dataset raw file")
        parser.add_argument("--src-lang", type=str,
                            help="source sentence language (postfix file name of source dataset)")
        parser.add_argument("--tgt-lang", type=str,
                            help="target sentence language (postfix file name of target dataset)")
        parser.add_argument("--update-dictionary", action="store_true", default=False,
                            help="update source and target dictionary")
        parser.add_argument("--max-words", type=int, default=30000,
                            help="max dictionary size")
        parser.add_argument("--freq-clip", type=int, default=5,
                            help="min word frequency")
        parser.add_argument("--max-train-pairs", type=int, default=10000,
                            help="max train dataset size")
        parser.add_argument("--max-eval-pairs", type=int, default=1000,
                            help="max eval dataset size")
        parser.add_argument("--logging-dir", type=str, help="logging directory")
        parser.add_argument("--num-workers", type=int, default=0,
                            help="number of workers to load dataset")
        parser.add_argument('--beam-size', type=int, default=1,
                            help='beam search size for generation')
        parser = cls.add_optimizer_args(parser)
        parser = cls.add_train_args(parser)
        parser = Transformer.add_args(parser)
        return parser

    @classmethod
    def add_optimizer_args(cls, parser):
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'radam', 'sgd'],
                            help='optimizer configuration')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='base learning rate')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='beta1 of adam-like optimizers')
        parser.add_argument('--beta2', type=float, default=0.98,
                            help='beta2 of adam-like optimizers')
        parser.add_argument('--weight-decay', type=float, default=1e-3,
                            help='weight decay of adam-like optimizers')
        parser.add_argument('--warmup-init-lr', type=float, default=1e-5,
                            help='warmup initialization learning rate')
        parser.add_argument('--warmup-steps', type=int, default=1000,
                            help='warmup steps')
        parser.add_argument('--momentum', type=float, default=0.0,
                            help='momentum of SGD optimizer')
        return parser

    @classmethod
    def add_train_args(cls, parser):
        parser.add_argument('--batch-size', type=int, default=256, help='batch size')
        parser.add_argument('--per-update-steps', type=int, default=1,
                            help='update steps for gradient accumulation')
        parser.add_argument('--clip-norm', type=float, default=100.0,
                            help='gradient norm clip')
        parser.add_argument('--test-per-train-epochs', type=int, default=1,
                            help='run test after N epochs')
        parser.add_argument('--epochs', type=int, default=100,
                            help='number of epochs')
        parser.add_argument('--fp16', action='store_true', default=False,
                            help='use mixed precision')
        return parser
    
    @classmethod
    def build_task(cls, args):
        source_dict, target_dict = cls.build_dictionary(args)
        train_dataloader, eval_dataloader = cls.build_dataloader(
            args, source_dict, target_dict)
        model = Transformer.build_model(args, source_dict, target_dict).cuda()
        optimizer, scheduler = cls.build_optimizer(args, model)
        # mixed precision
        if args.fp16:
            amp.initialize(model, optimizer, opt_level="O1")
        
        return cls(args, model, optimizer, scheduler, train_dataloader, eval_dataloader, source_dict, target_dict)
    
    @classmethod
    def build_task_from_config(cls, path):
        args = load_config_to_args(path)
        return cls.build_task(args)

    @classmethod
    def build_optimizer(cls, args, model):
        optimizer_cls = optimizer_map[args.optimizer]
        if args.optimizer == 'adam' or args.optimizer == 'radam':
            optimizer = optimizer_cls(model, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optimizer_cls(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported.")
        scheduler = InverseSquareRootScheduler(
            optimizer=optimizer,
            warmup_init_lr=args.warmup_init_lr,
            max_lr=args.lr,
            warmup_updates=args.warmup_steps,
        )
        return optimizer, scheduler

    @classmethod
    def build_dataloader(cls, args, source_dict, target_dict):
        assert args.per_update_steps > 0 and args.batch_size % args.per_update_steps == 0
        def _init_dataloader(source_file, target_file):
            sen_pair_dataset = Seq2SeqDataset(
                data_dir=args.data_dir,
                src_data_file=source_file,
                tgt_data_file=target_file,
                src_dict=source_dict,
                tgt_dict=target_dict,
                max_seq_length=args.max_seq_length,
                max_dataset_size=args.max_train_pairs
            )
            m_dataloader = Seq2SeqDataLoader(
                dataset=sen_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                padding_idx=source_dict.pad(),
                num_workers=args.num_workers,
                cuda=False # !!! load to CPU for better coding
            )
            return m_dataloader
        source_train_file = args.train_file + '.' + args.src_lang
        target_train_file = args.train_file + '.' + args.tgt_lang
        train_dataloader = _init_dataloader(source_train_file, target_train_file)
        source_eval_file = args.eval_file + '.' + args.src_lang
        target_eval_file = args.eval_file + '.' + args.tgt_lang
        eval_dataloader = _init_dataloader(source_eval_file, target_eval_file)
        return train_dataloader, eval_dataloader

    @classmethod
    def build_dictionary(cls, args):
        source_train_file = args.train_file + '.' + args.src_lang
        target_train_file = args.train_file + '.' + args.tgt_lang
        if args.share_encoder_decoder_embed:
            if args.update_dictionary:
                source_dict = target_dict = Dictionary.build_dictionary(
                    args.data_dir, [source_train_file, target_train_file],
                    max_words=args.max_words, threshold=args.freq_clip
                )
            else:
                source_dict = target_dict = Dictionary.load_dictionary(
                    args.data_dir, args.src_lang + '-' + args.tgt_lang + '.dict'
                )
        else:
            if args.update_dictionary:
                source_dict = Dictionary.build_dictionary(
                    args.data_dir, [source_train_file],
                    max_words=args.max_words, threshold=args.freq_clip
                )
                target_dict = Dictionary.build_dictionary(
                    args.data_dir, [target_train_file],
                    max_words=args.max_words, threshold=args.freq_clip
                )
            else:
                source_dict = Dictionary.load_dictionary(
                    args.data_dir, args.src_lang + '.dict'
                )
                target_dict = Dictionary.load_dictionary(
                    args.data_dir, args.tgt_lang + '.dict'
                )
        return source_dict, target_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`Dictionary`."""
        return self.tgt_dict

    def backward(self, loss):
        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def _multiply_grads(self, c):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(c)

    def train_step(self, samples):
        src_tokens, tgt_tokens = samples
        src_tokens_per_step = torch.chunk(src_tokens, chunks=self.args.per_update_steps, dim=0)
        tgt_tokens_per_step = torch.chunk(tgt_tokens, chunks=self.args.per_update_steps, dim=0)
        loss = 0.
        ntokens = 0
        # forward
        self.model.train()
        self.optimizer.zero_grad()
        for i, (src_tokens, tgt_tokens) in enumerate(zip(src_tokens_per_step, tgt_tokens_per_step)):
            src_tokens, tgt_tokens = src_tokens.cuda(), tgt_tokens.cuda()
            pred_logits = self.model(src_tokens, tgt_tokens)
            loss_info = self.model.compute_loss(pred_logits, tgt_tokens)
            # backward loss
            self.backward(loss_info['loss'])
            loss += loss_info['loss'].item()
            ntokens += loss_info['ntokens']
        # avgerage loss per token
        self._multiply_grads(1.0 / ntokens)
        # clip norm
        total_norm = clip_grad_norm_(
            self.model.parameters(), max_norm=self.args.clip_norm
        )
        # update params and change lr, zero grad
        self.optimizer.step()
        self.scheduler.step()
        return loss / ntokens, total_norm

    def valid_step(self, samples):
        src_tokens, tgt_tokens = samples
        src_tokens_per_step = torch.chunk(src_tokens, chunks=self.args.per_update_steps, dim=0)
        tgt_tokens_per_step = torch.chunk(tgt_tokens, chunks=self.args.per_update_steps, dim=0)
        loss = 0.
        ntokens = 0
        pred_sentences = []
        self.model.eval()
        with torch.no_grad():
            for i, (src_tokens, tgt_tokens) in enumerate(zip(src_tokens_per_step, tgt_tokens_per_step)):
                src_tokens, tgt_tokens = src_tokens.cuda(), tgt_tokens.cuda()
                pred_logits = self.model(src_tokens, tgt_tokens)
                loss_info = self.model.compute_loss(pred_logits, tgt_tokens)
                loss += loss_info['loss'].item()
                ntokens += loss_info['ntokens']
                pred_tokens, _ = self.model.get_normalized_probs(pred_logits, log_probs=False)
                pred_sentences.append(pred_tokens)
        pred_sentences = torch.cat(pred_sentences, dim=0)
        return loss / ntokens, pred_sentences

    def test_step(self, samples):
        src_tokens, tgt_tokens = samples
        src_tokens_per_step = torch.chunk(src_tokens, chunks=self.args.per_update_steps, dim=0)
        tgt_tokens_per_step = torch.chunk(tgt_tokens, chunks=self.args.per_update_steps, dim=0)
        pred_sentences = []
        self.model.eval()
        with torch.no_grad():
            for i, (src_tokens, tgt_tokens) in enumerate(zip(src_tokens_per_step, tgt_tokens_per_step)):
                src_tokens, tgt_tokens = src_tokens.cuda(), tgt_tokens.cuda()
                pred_tokens = self.model.generate_sentences(
                    src_tokens, tgt_tokens, beam=args.beam_size, teacher_forcing=False)
                pred_sentences.append(pred_tokens)
        pred_sentences = torch.cat(pred_sentences, dim=0)
        return pred_sentences

    def train_epoch(self):
        # checkpoint and logging
        for i, samples in tqdm(enumerate(self.train_dataloader)):
            loss, norm = self.train_step(samples)


    def valid_epoch(self):
        for i, samples in tqdm(enumerate(self.eval_dataloader)):
            src_tokens, tgt_tokens = samples
            loss, pred_tokens = self.valid_step(samples)
            # log loss

    def test_epoch(self):
        for i, samples in tqdm(enumerate(self.eval_dataloader)):
            pred_tokens = self.valid_step(samples)
            # log bleu and distinct

