import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.core import *
from jack.readers.classification.shared import MCAnnotation
from jack.readers.classification import util
from jack.util import preprocessing

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

class BertEmbeddingInputModule(OnlineInputModule[MCAnnotation]):
    
    def setup(self):
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.embeddings = self.shared_resources.embeddings
        # embeddings is only for compatibility, not used
        
        # Bert configuration
        layer_indexes = [-1, -2, -3, -4]
        layer_aggregation = 'sum'
        bert_model = 'bert-base-uncased'
        do_lower_case = True
        
        # Bert setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.layer_indexes = layer_indexes
        # TODO: add option for layer selection
        self.layer_aggregation = layer_aggregation
        # TODO: add option for layer aggregation
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model,
                do_lower_case=do_lower_case)
        # TODO: add option for tokenization

        self.model = BertModel.from_pretrained(bert_model)
        self.model.to(self.device)
        self.model.eval()
    
    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        # Same as ClassificationSingleSupportInputModule
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
        return [Ports.Input.emb_support, Ports.Input.emb_question,
                Ports.Input.support_length, Ports.Input.question_length,
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

        
    