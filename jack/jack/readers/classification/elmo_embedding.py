from typing import NamedTuple, List, Optional, Iterable, Tuple, Mapping
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.core import TensorPort, Ports, OnlineInputModule, QASetting, Answer, progressbar
from jack.readers.classification import util
from jack.util import preprocessing
from jack.util.map import numpify

from allennlp.modules.elmo import Elmo, batch_to_ids

MCAnnotation = NamedTuple('MCAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('answer', Optional[int]),
    ('id', Optional[int]),
])

# Original ELMo options & weight
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Small ELMo options & weight
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"


class ELMoEmbeddingInputModule(OnlineInputModule[MCAnnotation]):

    def setup(self):
        print("Setting up Elmo Embedding")
        self.vocab = self.shared_resources
        self.config = self.shared_resources.config
        self.embeddings = self.shared_resources.embeddings
        if self.embeddings is not None:
            self.__default_vec = np.zeros([self.embeddings.shape[-1]])
        
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        if torch.cuda.is_available():
            self.elmo.cuda()
        
    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        vocab = self.shared_resources.vocab
        if not vocab.frozen:
            preprocessing.fill_vocab(
                (q for q, _ in data), vocab, lowercase=self.shared_resources.config.get('lowercase', True))
            vocab.freeze()
        if not hasattr(self.shared_resources, 'answer_vocab') or not self.shared_resources.answer_vocab.frozen:
            self.shared_resources.answer_vocab = util.create_answer_vocab(
                qa_settings=(q for q, _ in data), answers=(a for _, ass in data for a in ass))
            self.shared_resources.answer_vocab.freeze()
        self.shared_resources.char_vocab = preprocessing.char_vocab_from_vocab(self.shared_resources.vocab)
    
    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]
    
    @property
    def output_ports(self) -> List[TensorPort]:
        if self.shared_resources.embeddings is not None:
            return [Ports.Input.emb_support, Ports.Input.emb_question,
                    Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    Ports.Input.sample_id,
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_batch_words, Ports.Input.support_batch_words,
                    Ports.is_eval]
        else:
            return [Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    Ports.Input.sample_id,
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_batch_words, Ports.Input.support_batch_words,
                    Ports.is_eval]
    
    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[MCAnnotation]:
        if answers is None:
            answers = [None] * len(questions)
        preprocessed = []
        if len(questions) > 1000:
            bar = progressbar.ProgressBar(
                max_value=len(questions),
                widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
            for i, (q, a) in bar(enumerate(zip(questions, answers))):
                preprocessed.append(self.preprocess_instance(i, q, a))
        else:
            for i, (q, a) in enumerate(zip(questions, answers)):
                preprocessed.append(self.preprocess_instance(i, q, a))
        
        return preprocessed
    
    def preprocess_instance(self, idd: int, question: QASetting,
                            answers: Optional[List[Answer]] = None) -> MCAnnotation:
        has_answers = answers is not None

        q_tokenized, q_ids, q_length, _, _ = preprocessing.nlp_preprocess(
            question.question, self.shared_resources.vocab,
            lowercase=self.shared_resources.config.get('lowercase', True))
        s_tokenized, s_ids, s_length, _, _ = preprocessing.nlp_preprocess(
            question.support[0], self.shared_resources.vocab,
            lowercase=self.shared_resources.config.get('lowercase', True))
        
        return MCAnnotation(
            question_tokens=q_tokenized,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_ids=s_ids,
            support_length=s_length,
            answer=self.shared_resources.answer_vocab(answers[0].text) if has_answers else 0,
            id=idd
        )
    
    def create_batch(self, annotations: List[MCAnnotation],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        word_chars, word_lengths, tokens, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(
                [a.question_tokens for a in annotations] + [a.support_tokens for a in annotations],
                self.shared_resources.char_vocab)
        question_words, support_words = tokens[:len(annotations)], tokens[len(annotations):]

        q_lengths = [a.question_length for a in annotations]
        s_lengths = [a.support_length for a in annotations]
        xy_dict = {
            Ports.Input.question_length: q_lengths,
            Ports.Input.support_length: s_lengths,
            Ports.Input.sample_id: [a.id for a in annotations],
            Ports.Input.word_chars: word_chars,
            Ports.Input.word_char_length: word_lengths,
            Ports.Input.question_batch_words: question_words,
            Ports.Input.support_batch_words: support_words,
            Ports.is_eval: is_eval,
            Ports.Input.support: [a.support_ids for a in annotations],
            Ports.Input.question: [a.question_ids for a in annotations]
        }
        if with_answers:
            xy_dict[Ports.Target.target_index] = [a.answer for a in annotations]
        xy_dict = numpify(xy_dict)

        # Elmo embeddings
        tokens_support = [a.support_tokens for a in annotations]
        tokens_question = [a.question_tokens for a in annotations]
        chars_support = batch_to_ids(tokens_support)
        chars_question = batch_to_ids(tokens_question)

        if torch.cuda.is_available():
            chars_support = chars_support.cuda()
            chars_question = chars_question.cuda()

        with torch.no_grad():
            emb_support = self.elmo(chars_support)['elmo_representations'][0].detach()
            emb_question = self.elmo(chars_question)['elmo_representations'][0].detach()
        
        xy_dict[Ports.Input.emb_support] = emb_support
        xy_dict[Ports.Input.emb_question] = emb_question

        return xy_dict