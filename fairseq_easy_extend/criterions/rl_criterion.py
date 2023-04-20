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
        batch_size, sent_len, vocab_size = outputs.size()[0], outputs.size()[1], outputs.size()[2]
        
        #softmax outputs
        outputs_prob = F.softmax(outputs, dim=-1).view(-1, vocab_size)
        
        #multinomial sampling 
        sample_sent_idx = torch.multinomial(outputs_prob, 1, True).view(batch_size, sent_len)
        
        #convert to string sentence
        sample_sent_str = [self.tgt_dict.string(sample) for sample in sample_sent_idx]
        target_sent_str = [self.tgt_dict.string(target).replace("<pad>", "").strip() for target in targets]        
        # print(sample_sent_str)
        # print(target_sent_str)
        
        #compute evaluation score
        R = self.compute_risk(sample_sent_str, target_sent_str, sent_len)
        R = R.to(outputs.device)
        
        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
            sample_sent_idx, R = sample_sent_idx[masks], R[masks]
            
        print(outputs.size(), sample_sent_idx.size())
        
        outputs_logprob = F.log_softmax(outputs, dim=-1)
        sample_logprob = torch.gather(outputs_logprob, dim=-1, index=sample_sent_idx.view(-1,1)).squeeze(-1)
        loss = -sample_logprob*R
        loss = loss.mean()
        print(loss)
        #compute repetition
        self.repetition = self.compute_repetition(sample_sent_str, target_sent_str)
        return loss

    ## Calculate reward
    def compute_risk(self, sampled_sentences, targets, sent_len):
        with torch.no_grad():
            if self.metric == "CHRF":
                chrf = CHRF()
                R = torch.tensor([chrf.corpus_score([sample], [[target]]).score for sample,target in zip(sampled_sentences,targets)])
                R = R.repeat(sent_len, 1).T

            elif self.metric == "COMET":
                ter = TER()
                R = torch.tensor([ter.corpus_score(pred, target).score for pred, target in zip(preds, targets)])
        return R

    ## sample
#     def sampling(self, outputs, sample_type:str="argmax", n:int=1):
# #         #softmax over outputs
# #         soft_max = torch.nn.Softmax(dim=-1)
# #         outputs_softmax = soft_max(outputs)     
#         if sample_type == "argmax":
#             #argmax over softmax 
#             outputs_ids = torch.argmax(outputs,dim=-1)
#             outputs_prob = outputs.max(dim=-1).values
#             log_prob = torch.log(outputs_prob)
#             log_prob = torch.sum(log_prob, dim=-1)
            
#         else:
#             #multinomial sampling
#             outputs_ids = torch.multinomial(outputs, n, True)
#             log_prob = torch.sum(torch.log(torch.gather(outputs, dim=-1, index=outputs_ids).T),dim=-1)
#             print(log_prob)
#             outputs_ids = outputs_ids.T
#         return log_prob, outputs_ids


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

    def compute_repetition(self, sample_sentences, targets):
        repetition_ratio = [(len(sample.split())-len(set(sample.split())))/(len(target.split())-len(set(target.split()))) for sample,target in zip(sample_sentences,targets)]
        return sum(repetition_ratio)/len(repetition_ratio)

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
