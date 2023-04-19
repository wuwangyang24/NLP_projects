import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
from sacrebleu.metrics import BLEU, CHRF, TER

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
        print(f"metric: {self.metric}")

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """
        
        #softmax outputs
        soft_max = torch.nn.Softmax(dim=-1)
        outputs = soft_max(outputs)
        
        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
        print(outputs.size())
        print(targets.size())
        
        outputs_prob, outputs_ids = self.sampling(outputs, "argmax")
        #convert to string sentence
        sampled_sentence = self.tgt_dict.string(outputs_ids)
        targets = self.tgt_dict.string(targets)
        print(f"sampled sentence: {sampled_sentence}")
        print(f"target sentence: {targets}")
        #compute loss
        R = self.compute_risk([sampled_sentence], [[targets]])
        print(f"R:{R}")
        print(f"log:{-self.log_prob(outputs_prob)}")
        loss = -self.log_prob(outputs_prob)*R

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
    def compute_risk(self, sampled_sentence, targets):
      with torch.no_grad():
        if self.metric == "CHRF":
          chrf = CHRF()
          R = chrf.corpus_score(sampled_sentence, targets).score
          R = 100-R
        elif self.metric == "COMET":
          ter = TER()
          R = torch.tensor([ter.corpus_score(pred, target).score for pred, target in zip(preds, targets)])
      return R

    ## Compute the log probability of outputs 
    def log_prob(self, outputs_prob):
      log_prob = torch.log(outputs_prob)
      log_prob = torch.sum(log_prob, dim=-1)
      return log_prob

    ## sample
    def sampling(self, outputs, sample_type:str="argmax", n:int=1):
#         #softmax over outputs
#         soft_max = torch.nn.Softmax(dim=-1)
#         outputs_softmax = soft_max(outputs)     
        if sample_type == "argmax":
            #argmax over softmax 
            outputs_ids = torch.argmax(outputs,dim=-1)
            outputs_prob = outputs.max(dim=-1).values
        else:
            #multinomial sampling
            outputs_ids = torch.multinomial(outputs, n, True)
            outputs_prob = torch.tensor([torch.gather(outputs, dim=-1, indices=outputs_multinomial[:,col].unsqueeze(-1)).squeeze(-1) for col in range(n)])
        return outputs_prob, outputs_ids


    def forward(self, model, samples, reduce=True):
        outputs = model(samples['net_input']['src_tokens'], 
                      samples['net_input']['src_lengths'], 
                      samples['prev_target'], 
                      samples['target'])
        targets = samples['target']
        loss = self._compute_loss(outputs['word_ins']['out'], 
                                targets, 
                                masks=outputs['word_ins'].get('mask',None),
                                # label_smoothing=outputs['word_ins']['ls'],
                                # factor=outputs['length']['factor']
        )
        repetition = self.compute_repetition(outputs['word_ins']['out'], outputs['word_ins'].get('mask',None))
        nsentences = samples['nsentences']
        ntokens = samples['ntokens']
        sample_size = 1
        outputs_logging = {
                         'loss': loss.detach(),
                         'nsentences':nsentences, 
                         'ntokens': ntokens,
                         'sample_size': sample_size,
                         'repetition': repetition/nsentences
                         }
        print(f"Loss: {loss}")
        print(f"repetition: {repetition/nsentences}")
        return loss, sample_size, outputs_logging

    def compute_repetition(self, outputs, masks):
        if masks is not None:
            outputs = outputs[masks]
        outputs_softmax,outputs_argmax = self.sampling(outputs)
        sampled_sentence = self.tgt_dict.string(outputs_argmax)
        repetition = len(sampled_sentence.split())-len(set(sampled_sentence.split()))
        return repetition

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]]) -> None:  
        """Aggregate logging outputs from data parallel training."""
        loss = [log.get("loss", 0) for log in logging_outputs]
        loss = sum(loss)/len(loss)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        repetition = [log.get("repetition", 0) for log in logging_outputs]
        repetition = sum(repetition)/len(repetition)
        
        print(logging_outputs)
        metrics.log_scalar("loss", loss)
        metrics.log_scalar('repetition', repetition)

