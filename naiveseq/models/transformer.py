import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from .modules import TransformerEncoder, TransformerDecoder, Embed2Token
from .criterions import LabelSmoothedCrossEntropyLoss
from .utils import load_config_to_args


class Transformer(nn.Module):

    def __init__(self, args, encoder, decoder, embed2token, criterion):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.embed2token = embed2token
        self.criterion = criterion

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='transformer embedding dimension')
        parser.add_argument('--ffn-embed-dim', type=int, metavar='N',
                            help='transformer inner dimension for FFN')
        parser.add_argument('--attention-heads', type=int, metavar='H',
                            help='num of attention heads')
        parser.add_argument('--learned-pos', action='store_true',
                            default=True, help='use learned positional embeddings')
        parser.add_argument('--normalize-before', action='store_true',
                            default=False, help='apply layernorm before each block')
        parser.add_argument('--layers', type=int, metavar='L',
                            help='num of layers')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            default=False, help='share decoder input and output embeddings')
        parser.add_argument('--share-encoder-decoder-embed', action='store_true',
                            default=True, help='share decoder and encoder embeddings')
        parser.add_argument('--activation-fn', default='relu',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use in FFN and Residual Block')
        parser.add_argument('--positional-attention', action='store_true',
                            default=False, help='use positional attention module in decoder')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            default=0.0, help='dropout probability for attention weights')
        parser.add_argument('--ffn-dropout', type=float, metavar='D',
                            default=0.1, help='dropout probability for FFN')
        parser.add_argument('--embed-dropout', type=float, metavar='D',
                            default=0.1, help='dropout probability after input embedding layer')
        parser.add_argument('--self-attn-policy', choices=['none', 'mask-future', 'mask-self'],
                            default='none', help='mask policy in decoder self attention module')
        parser.add_argument('--residual-policy', choices=['none', 'residual', 'highway'],
                            default='residual', help='residual policy in transformer')
        parser.add_argument('--residual-dropout', type=float, metavar='D',
                            default=0.1, help='dropout probability in residual block')
        parser.add_argument('--max-seq-length', type=int, metavar='L',
                            default=128, help='max sentence length')
        parser.add_argument('--label-smoothing', type=float,
                            default=0.1, help='label smoothing ratio in CrossEntropyLoss')
        return parser

    @classmethod
    def build_embedding(cls, args, dictionary):
        return nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=args.embed_dim,
            padding_idx=dictionary.pad(),
        )

    @classmethod
    def build_encoder(cls, args, dictionary, embedding: Optional[nn.Embedding] = None):
        return TransformerEncoder(
            vocab_size=len(dictionary),
            embed_dim=args.embed_dim,
            token_embedding=embedding,
            padding_idx=dictionary.pad(),
            num_layers=args.layers,
            learned_pos_embedding=args.learned_pos,
            embed_dropout=args.embed_dropout,
            num_heads=args.attention_heads,
            attn_dropout=args.attention_dropout,
            attn_bias=True,
            ffn_embed_dim=args.ffn_embed_dim,
            ffn_dropout=args.ffn_dropout,
            ffn_activation_fn=args.activation_fn,
            residual_policy=args.residual_policy,
            residual_dropout=args.residual_dropout,
            max_seq_length=args.max_seq_length,
            post_norm=(not args.normalize_before),
        )

    @classmethod
    def build_decoder(cls, args, dictionary, embedding: Optional[nn.Embedding] = None):
        return TransformerDecoder(
            vocab_size=len(dictionary),
            embed_dim=args.embed_dim,
            token_embedding=embedding,
            padding_idx=dictionary.pad(),
            num_layers=args.layers,
            learned_pos_embedding=args.learned_pos,
            embed_dropout=args.embed_dropout,
            num_heads=args.attention_heads,
            attn_dropout=args.attention_dropout,
            attn_bias=True,
            self_attn_policy=args.ffn_embed_dim,
            ffn_embed_dim=args.ffn_dropout,
            ffn_dropout=args.ffn_dropout,
            ffn_activation_fn=args.activation_fn,
            residual_policy=args.residual_policy,
            residual_dropout=args.residual_dropout,
            max_seq_length=args.max_seq_length,
            post_norm=(not args.normalize_before),
            apply_pos_attn=args.positional_attention,
        )

    @classmethod
    def build_criterion(cls, args, dictionary):
        return LabelSmoothedCrossEntropyLoss(
            eps=args.label_smoothing,
            ignore_index=dictionary.pad(),
            reduction='sum',
        )

    @classmethod
    def build_embed2token(cls, args, dictionary, embedding: Optional[nn.Embedding] = None):
        if args.share_decoder_input_output_embed:
            assert embedding
            return Embed2Token(
                vocab_size=len(dictionary),
                embed_dim=args.embed_dim,
                embedding_layer=embedding,
                bias=True,
            )
        else:
            return Embed2Token(
                vocab_size=len(dictionary),
                embed_dim=args.embed_dim,
                embedding_layer=None,
                bias=True,
            )

    @classmethod
    def build_model(cls, args, source_dictionary, target_dictionary):
        src_dict, tgt_dict = source_dictionary, target_dictionary
        encoder_embedding = cls.build_embedding(args, src_dict)
        if args.share_encoder_decoder_embed:
            if src_dict != tgt_dict:
                raise ValueError("--share-encoder-decoder-embed requires a joined dictionary")
            decoder_embedding = encoder_embedding
        else:
            decoder_embedding = cls.build_embedding(args, tgt_dict)

        encoder = cls.build_encoder(args, src_dict, encoder_embedding)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embedding)
        embed2token = cls.build_embed2token(args, tgt_dict, decoder_embedding)
        criterion = cls.build_criterion(args, tgt_dict)
        return cls(args, encoder, decoder, embed2token, criterion)

    @classmethod
    def build_model_from_config(cls, path, task):
        args = load_config_to_args(path)
        return cls.build_model(args, task)

    def forward(self, src_tokens, tgt_tokens):
        enc_feats, enc_inner_states, sentence_rep, src_padding_mask = self.encoder(src_tokens)
        tgt_tokens = tgt_tokens[:, :-1]
        dec_feats, dec_inner_states = self.decoder(tgt_tokens, enc_feats, src_padding_mask)
        pred_logits = self.embed2token(dec_feats)
        # pred_lprobs = self.embed2token.get_normalized_probs(pred_logits)
        return pred_logits

    def compute_loss(self, output_logits, tgt_tokens):
        """
        return: {
            'loss': loss,
            'ntokens': ntokens,
        }
        """
        tgt_tokens = tgt_tokens[:, 1:]
        return self.criterion(output_logits, tgt_tokens)

    def get_normalized_probs(self, output_logits, log_probs=True):
        return self.embed2token.get_normalized_probs(
            output_logits, log_probs=log_probs)

    def get_sentences(self, output_logits):
        max_values, max_indexes = output_logits.max(axis=-1)
        return max_values, max_indexes

    @torch.no_grad()
    def generate_sentences(self, src_tokens, tgt_tokens, beam: int=1, teacher_forcing: bool=False):
        enc_feats, enc_inner_states, sentence_rep, src_padding_mask = self.encoder(src_tokens)
        L, B, E = enc_feats.size
        # enc_feats: L x B x E, src_padding_mask: B x L
        if teacher_forcing:
            tgt_tokens = tgt_tokens[:, :-1]
            dec_feats, dec_inner_states = self.decoder(tgt_tokens, enc_feats, src_padding_mask)
            pred_logits = self.embed2token(dec_feats)
            pred_tokens = self.get_sentences(pred_logits)
            return pred_tokens
        else:
            finished = torch.BoolTensor([False for _ in range(B)]).to(src_tokens)
            all_finished = False
            max_len = 0
            L = self.args.max_seq_length
            V = len(self.tgt_dict)

            prefix_tgt_tokens = torch.zeros(B, beam, L+1).to(src_tokens).long().fill_(self.tgt_dict.pad())
            prefix_tgt_tokens[:, :, 0] = self.tgt_dict.bos()
            scores = torch.zeros(B, beam, L).to(src_tokens).float()
            final_length = torch.ones(B).to(src_tokens).long()

            enc_feats = enc_feats.repeat(1, beam, 1)
            src_padding_mask = src_padding_mask.repeat(beam, 1)
            
            while not all_finished and max_len < self.args.max_seq_length:
                decoder_input_tokens = prefix_tgt_tokens.view(B*beam, -1)[:, :max_len+1]
                for i in range(B):
                    decoder_input_tokens[i, :, final_length[i]:] = self.tgt_dict.pad()
                
                dec_feats, dec_inner_states = self.decoder(decoder_input_tokens, enc_feats, src_padding_mask)
                pred_logits = self.embed2token(dec_feats)
                pred_lprobs = self.embed2token(pred_logits, log_probs=True)
                
                pred_lprobs = pred_lprobs.view(
                    B, beam, max_len+1, -1)[:, :, max_len, :].squeeze()  # B x beam_size x V
                pred_lprobs[:, :, self.tgt_dict.pad()] = -math.inf  # never select pad
                pred_lprobs[:, :, self.tgt_dict.unk()] = -math.inf  # never select unk
                if max_len == 0:
                    pred_lprobs = pred_lprobs[:, ::beam, :].contiguous()  # B x 1 x V
                else:
                    assert scores is not None
                    pred_lprobs = pred_lprobs + scores[:, :, max_len-1].unsqueeze()  # B x beam_size x V
                
                pred_topk_lprobs, pred_topk_indices = torch.topk(
                    pred_lprobs.view(B, -1), # B x (beam_size*V) or B x V
                    k=beam,
                    dim=-1
                ) # shape: B x beam_size
                cand_scores = pred_topk_lprobs # shape: B x beam_size
                cand_beams = pred_topk_indices // V # shape: B x beam_size, [0, beam)
                cand_indices = pred_topk_indices.fmod(V)  # shape: B x beam_size, [0, V)
                
                # solve beam search core algo.
                prefix_tgt_tokens[:, :, max_len] = torch.gather(
                    prefix_tgt_tokens[:, :, max_len],
                    dim=1,
                    index=cand_beams
                )
                prefix_tgt_tokens[:, :, max_len+1] = cand_indices
                scores[:, :, max_len] = cand_scores

                finished = (cand_indices[:, 0] == self.tgt_dict.eos())
                max_len += 1
                final_length[~finished] += 1
                all_finished = (finished == True)
            return prefix_tgt_tokens[:, 0, :].squeeze(), final_length
