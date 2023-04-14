
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
import nltk
from sacrebleu.metrics import BLEU
from sacrebleu.metrics import TER

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="BLEU",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tgt_dict = task.target_dictionary

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        # #softmax over outputs
        # soft_max = torch.nn.Softmax(dim=-1)
        # outputs_softmax = soft_max(outputs)
        # #argmax over softmax 
        # outputs_argmax = torch.argmax(outputs_softmax,dim=-1)
        outputs_softmax,outputs_argmax = self.sampling(outputs)
        #convert to string sentence
        sampled_sentences = [self.tgt_dict.string(sentence) for sentence in outputs_argmax]
        targets = [self.tgt_dict.string(sentence) for sentence in targets]
        targets = [[sentence] for sentence in targets]
        #compute loss
        R = self.compute_reward(self.metric, sampled_sentences, targets)
        loss = torch.mul(-self.log_prob(outputs_softmax),R)

        #argmax over the softmax \ sampling (e.g. multinomial)
        #sampled_sentence = [4, 17, 18, 19, 20]
        #sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        #target_sentence = "I am a sentence"
        #with torch.no_grad()
            #R(*) = eval_metric(sampled_sentence_string, target_sentence)
            #R(*) is a number, BLEU, сhrf, etc.
        #loss = -log_prob(outputs)*R()

        loss = loss.mean()
        print(loss)
        return loss

    ## Calculate reward
    def compute_reward(self, sentence_level_metric, preds, targets):
      with torch.no_grad():
        if self.metric == "BLEU":
          bleu = BLEU()
          R = torch.tensor([bleu.corpus_score(pred, target).score for pred, target in zip(preds, targets)])
          R = 100/(R+1)
        elif self.metric == "TER":
          ter = TER()
          R = torch.tensor([ter.corpus_score(pred, target).score for pred, target in zip(preds, targets)])
      return R

    ## Compute the log probability of outputs 
    def log_prob(self, outputs):
      outputs_prob = torch.max(outputs,dim=-1)
      log_prob = torch.log(outputs_prob.values)
      log_prob = torch.sum(log_prob, dim=-1)
      return log_prob

    ## sample using argmax
    def sampling(self,outputs):
      #softmax over outputs
      soft_max = torch.nn.Softmax(dim=-1)
      outputs_softmax = soft_max(outputs)
      #argmax over softmax 
      outputs_argmax = torch.argmax(outputs_softmax,dim=-1)
      return outputs_softmax,outputs_argmax


    def forward(self, model, samples, reduce=True):
      outputs = model(samples['net_input']['src_tokens'], 
                      samples['net_input']['src_lengths'], 
                      samples['net_input']['prev_output_tokens'], 
                      samples['target'])
      targets = samples['target']
      loss = self._compute_loss(outputs['word_ins']['out'], 
                                targets, 
                                # masks=outputs['word_ins']['mask'],
                                label_smoothing=outputs['word_ins']['ls'],
                                # factor=outputs['length']['factor']
      )
      nsentences = samples['nsentences']
      ntokens = samples['ntokens']
      sample_size = nsentences*samples['net_input']['src_lengths'][0]
      outputs_logging = {
                         'loss': loss,
                         'nsentences':nsentences, 
                         'ntokens': ntokens,
                         'sample_size': sample_size,
                        #  'repetition_ratio': self.compute_repetition(outputs['word_ins']['out'])
                         }
      return loss, sample_size, outputs_logging

    def compute_repetition(self, outputs):
      outputs_softmax,outputs_argmax = self.sampling(outputs)
      sampled_sentences = [self.tgt_dict.string(sentence) for sentence in outputs_argmax]
      print(sampled_sentences[0])
      repetition_ratio = sum([len(tuple(sentence.split()))-len(sentence.split()) for sentence in sampled_sentences])/sum([len(sentence) for sentence in sampled_sentences])
      return repetition_ratio

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]]) -> None:  
      print(logging_outputs)
      """Aggregate logging outputs from data parallel training."""
      loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
      nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
      ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
      sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
      # repetition_ratio_sum = sum(log.get("repetition_ratio", 0) for log in logging_outputs)

      metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

      if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
      else:
          metrics.log_derived(
              "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
          )

      # metrics.log_scalar('repetition_ratio', repetition_ratio_sum/len(logging_outputs))
