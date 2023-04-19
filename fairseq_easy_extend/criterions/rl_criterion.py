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
        self.repetition = None

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
        
        log_prob, outputs_ids = self.sampling(outputs, "multinomial",n=5)
        #convert to string sentence
        sampled_sentence = [self.tgt_dict.string(sample) for sample in outputs_ids]
        targets = self.tgt_dict.string(targets)
        print(f"sampled sentence: {sampled_sentence}")
        print(f"target sentence: {targets}")
        #compute risk
        R = torch.tensor([self.compute_risk([sentence], [[targets]]) for sentence in sampled_sentence])
        print(f"R:{R}")
        loss = -log_prob*R
        loss = loss.mean()
        
        #compute repetition
        self.repetition = self.compute_repetition(sampled_sentence, targets)

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

    ## sample
    def sampling(self, outputs, sample_type:str="argmax", n:int=1):
#         #softmax over outputs
#         soft_max = torch.nn.Softmax(dim=-1)
#         outputs_softmax = soft_max(outputs)     
        if sample_type == "argmax":
            #argmax over softmax 
            outputs_ids = torch.argmax(outputs,dim=-1)
            outputs_prob = outputs.max(dim=-1).values
            log_prob = torch.log(outputs_prob)
            log_prob = torch.sum(log_prob, dim=-1)
            
        else:
            #multinomial sampling
            outputs_ids = torch.multinomial(outputs, n, True)
            log_prob = torch.sum(torch.log(torch.gather(outputs, dim=-1, index=outputs_ids).T),dim=-1)
            print(log_prob)
            outputs_ids = outputs_ids.T
        return log_prob, outputs_ids


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
        nsentences = samples['nsentences']
        ntokens = samples['ntokens']
        sample_size = 1
        outputs_logging = {
                         'loss': loss.detach(),
                         'nsentences':nsentences, 
                         'ntokens': ntokens,
                         'sample_size': sample_size,
                         'repetition': self.repetition
                         }
        print(f"Loss: {loss}")
        print(f"repetition: {self.repetition}")
        return loss, sample_size, outputs_logging

    def compute_repetition(self, sample_sentence, targets):
        repetition_sample = sum([len(sentence.split())-len(set(sentence.split())) for sentence in sample_sentence])/len(sample_sentence)
        repetition_target = len(targets.split())-len(set(targets.split()))
        return repetition_sample-repetition_target

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
