# -*- coding: utf-8 -*-

"""
Here we define light data structures to store the input to jack readers, and their output.
"""

from typing import Tuple, Sequence
import sys

class Answer:
    """
    Representation of an answer to a question.
    """

    def __init__(self, text: str, span: Tuple[int, int] = None, doc_idx: int = 0, score: float = 1.0):
        """
        Create a new answer.
        Args:
            text: The text string of the answer.
            span: For extractive QA, a span in the support documents. The triple `(start, end)`
                represents a span in support document with index `doc_index` in the ordered sequence of
            doc_idx: index of document where answer was found
            support documents. The span starts at `start` and ends at `end` (exclusive).
            score: the score a model associates with this answer.
        """
        assert span is None or len(span) == 2, "span should be (char_start, char_end) tuple"

        self.score = score
        self.span = span
        self.doc_idx = doc_idx
        self.text = text


class QASetting:
    """
    Representation of a single question answering problem. It primarily consists of a question,
    a list of support documents, and optionally, some set of candidate answers.
    """

    def __init__(self,
                 question: str,
                 support: Sequence[str] = (),
                 id: str = None,
                 candidates: Sequence[str] = None,
                 seq_candidates: Sequence[str] = None,
                 candidate_spans: Sequence[Tuple[int, int, int]] = None,
                 q_tokenized: Sequence[str] = None,
                 s_tokenized: Sequence[str] = None,
                 q_dep_i: Sequence[int] = None,
                 q_dep_j: Sequence[int] = None,
                 q_dep_type: Sequence[int] = None,
                 s_dep_i: Sequence[int] = None,
                 s_dep_j: Sequence[int] = None,
                 s_dep_type: Sequence[int] = None):
        """
        Create a new QASetting.
        Args:
            question: the question text.
            support: a sequence of support documents the answerer has access to when answering the question.
            id: an identifier for this question setting.
            candidates: a list of candidate answer strings.
            candidate_spans: for extractive QA, a sequence of candidate spans in the support documents.
            A span `(doc_index,start,end)` corresponds to a span in support document with index `doc_index`,
            with start position `start` and end position `end`.
        """
        self.id = id
        self.candidate_spans = candidate_spans
        self.candidates = candidates
        self.support = support
        self.question = question

        # Added for tokenization
        self.q_tokenized = q_tokenized
        self.s_tokenized = s_tokenized
        # Added for dependency information
        self.q_dep_i = q_dep_i
        self.q_dep_j = q_dep_j
        self.q_dep_type = q_dep_type
        self.s_dep_i = s_dep_i
        self.s_dep_j = s_dep_j
        self.s_dep_type = s_dep_type


def _jack_to_qasetting(instance, value, global_candidates):
    support = [value(s) for s in instance["support"]] if "support" in instance else None
    idd = value(instance, 'id')
    for question_instance in instance["questions"]:
        question = value(question_instance['question'])
        idd = value(question_instance, 'id') or idd
        idd = value(question_instance['question'], 'id') or idd
        if global_candidates is None:
            candidates = [value(c) for c in question_instance['candidates']] if "candidates" in question_instance else None
        else:
            candidates = global_candidates
        answers = [Answer(value(c), value(c, 'span'), value(c, 'doc_idx', 0)) for c in
                   question_instance['answers']] if "answers" in question_instance else None
        if "q_tokenized" in instance:
            yield QASetting(question, support, candidates=candidates, id=idd,
                q_tokenized=instance['q_tokenized'],
                q_dep_i=instance['q_dep_i'],
                q_dep_j=instance['q_dep_j'],
                q_dep_type=instance['q_dep_type'],
                s_tokenized=instance['s_tokenized'],
                s_dep_i=instance['s_dep_i'],
                s_dep_j=instance['s_dep_j'],
                s_dep_type=instance['s_dep_type']
                ), answers
        else:
            yield QASetting(question, support, candidates=candidates, id=idd), answers


def jack_to_qasetting(jtr_data, max_count=None):
    """
    Converts a python dictionary in Jack format to a QASetting.
    Args:
        jtr_data: dictionary extracted from jack json file.
        max_count: maximal number of instances to load.

    Returns:
        list of QASetting
    """

    def value(c, key="text", default=None):
        # return c.get(key, default) if isinstance(c, dict) else c if key == 'text' else default
        if isinstance(c, dict):
            return c.get(key, default)
        elif key == 'text':
            return c
        else:
            return default

    global_candidates = [value(c) for c in jtr_data['globals']['candidates']] if 'globals' in jtr_data else None

    # # filter None
    # cnt_none = 0
    # instances = []
    # for instance in jtr_data["instances"]:
    #     if "q_tokenized" in instance:
    #         if (None in instance['q_dep_i']) or (None in instance['q_dep_j']) or (None in instance['q_dep_type'])\
    #             or (None in instance['s_dep_i']) or (None in instance['s_dep_j']) or (None in instance['s_dep_type']):
    #             cnt += 1
    #             continue
    #         else:
    #             instances.append(instance)
    #     else:
    #         instances = jtr_data["instances"]
    #         break
    # print('Filter %d None instances' % cnt, file=sys.stderr)

    ans = [(inp, answer) for i in jtr_data["instances"]
           for inp, answer in _jack_to_qasetting(i, value, global_candidates)][:max_count]
    return ans
