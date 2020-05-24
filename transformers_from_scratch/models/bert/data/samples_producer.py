import json
import pathlib
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Process, Queue
from typing import List, Sequence, Tuple, Generator

import numpy as np
from tokenizers import BertWordPieceTokenizer, Encoding


class _SentencesChunk:
    @property
    def size(self):
        return self._size

    @property
    def sentences(self):
        return self._chunk

    @property
    def doc_n_sents(self):
        return self._doc_n_sents

    def __init__(self):
        self._chunk = []
        self._doc_n_sents = []
        self._size = 0

    def add_sentences(self, sentences: Sequence[str]):
        self._doc_n_sents.append(len(sentences))
        self._chunk.extend(sentences)
        self._size += len(sentences)

    def add_sentences_from_line(self, line: str):
        sentences = json.loads(line)['sentences']
        self.add_sentences(sentences)


@dataclass
class Sample:
    token_ids: List[int]
    labels: List[int]
    token_type_ids: List[int]
    is_distractor: bool


class SamplesProducer(Process):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: BertWordPieceTokenizer,
            tokenization_chunk_size: int,
            max_len: int,
            masked_lm_prob: float,
            out_samples_queue: Queue
    ):
        super().__init__(daemon=True)

        self._file_path = file_path
        self._tokenizer = tokenizer
        self._tokenization_chunk_size = tokenization_chunk_size
        self._max_len = max_len
        self._masked_lm_prob = masked_lm_prob
        self._out_samples_queue = out_samples_queue

        self._mask_token_id = self._tokenizer.token_to_id('[MASK]')
        self._cls_token_id = self._tokenizer.token_to_id('[CLS]')
        self._sep_token_id = self._tokenizer.token_to_id('[SEP]')

        assert isinstance(self._mask_token_id, int)
        assert isinstance(self._cls_token_id, int)
        assert isinstance(self._sep_token_id, int)

    def run(self) -> None:

        with self._file_path.open() as file:

            sentences_chunk = _SentencesChunk()

            for line in cycle(file):
                sentences_chunk.add_sentences_from_line(line=line)

                if sentences_chunk.size >= self._tokenization_chunk_size:
                    doc_sent_encodings = self._encode_sentences_chunk(
                        chunk=sentences_chunk
                    )

                    samples_generator = self._get_samples_from_encodings(
                        doc_sent_encodings=doc_sent_encodings
                    )

                    for sample in samples_generator:
                        self._out_samples_queue.put(sample)

                    sentences_chunk = _SentencesChunk()

    def _get_samples_from_encodings(
            self,
            doc_sent_encodings: Sequence[Sequence[Encoding]]
    ) -> Generator[Sample, None, None]:
        for doc in doc_sent_encodings:

            if len(doc) == 0:
                continue

            if len(doc) == 1 or np.random.rand() > 0.5:
                distr_doc = np.random.choice(doc_sent_encodings)
                distr_sent = np.random.choice(distr_doc)
            else:
                distr_sent = None

            seqs_a = []
            seq_a_len = 0
            for i_sent, sent in enumerate(doc[:-1]):

                if distr_sent is not None:
                    seq_b = distr_sent
                else:
                    seq_b = doc[i_sent + 1]

                seqs_a.append(sent)
                seq_a_len += len(sent.ids)

                if (seq_a_len + len(seq_b) >= self._max_len - 3) or \
                        (i_sent == len(doc) - 2):
                    sample = self._get_sample_from_sequences(
                        seqs_a=seqs_a,
                        seq_b=seq_b,
                        is_distractor=distr_sent is not None
                    )

                    seqs_a = []
                    seq_a_len = 0

                    yield sample

    def _get_sample_from_sequences(
            self,
            seqs_a: Sequence[Encoding],
            seq_b: Encoding,
            is_distractor: bool
    ) -> Sample:
        words_a = _get_words_from_sequences(sequences=seqs_a)
        words_b = _get_words_from_sequences(sequences=[seq_b])

        seq_a_token_ids, seq_a_labels = self._get_ids_and_labels_from_words(
            words=words_a
        )
        seq_b_token_ids, seq_b_labels = self._get_ids_and_labels_from_words(
            words=words_b
        )

        crops_a, crops_b = self._get_crops(
            seq_len_a=len(seq_a_token_ids),
            seq_len_b=len(seq_b_token_ids)
        )

        a_token_ids = seq_a_token_ids[slice(*crops_a)]
        b_token_ids = seq_b_token_ids[slice(*crops_b)]
        a_labels = seq_a_labels[slice(*crops_a)]
        b_labels = seq_b_labels[slice(*crops_b)]

        token_ids = [self._cls_token_id] + a_token_ids + \
                    [self._sep_token_id] + b_token_ids + \
                    [self._sep_token_id]

        labels = [-100] + a_labels + \
                 [-100] + b_labels + \
                 [-100]

        token_type_ids = [0] + [0] * len(a_labels) + \
                         [0] + [1] * len(b_labels) + \
                         [1]

        sample = Sample(
            token_ids=token_ids,
            labels=labels,
            token_type_ids=token_type_ids,
            is_distractor=is_distractor
        )

        return sample

    def _get_crops(
            self,
            seq_len_a: int,
            seq_len_b: int
    ) -> Tuple[List[int], List[int]]:
        max_len = min(self._max_len - 3, seq_len_a + seq_len_b)
        if np.random.rand() < 0.1 and max_len > 2:
            max_len = np.random.randint(2, max_len)

        crops_a = [0, seq_len_a]
        crops_b = [0, seq_len_b]

        i = 0
        while True:
            if max_len >= (seq_len_a + seq_len_b):
                return crops_a, crops_b
            elif seq_len_a > seq_len_b:
                seq_len_a -= 1
                crops = crops_a
            else:
                seq_len_b -= 1
                crops = crops_b

            if i % 2:
                crops[1] -= 1
            else:
                crops[0] += 1

            i += 1

    def _encode_sentences_chunk(
            self,
            chunk: _SentencesChunk
    ) -> List[List[Encoding]]:
        sent_encodings = self._tokenizer.encode_batch(
            sequences=chunk.sentences,
            add_special_tokens=False
        )

        doc_sent_encodings = []

        for n_sents in chunk.doc_n_sents:
            doc_sent_encodings.append(sent_encodings[:n_sents])
            sent_encodings = sent_encodings[n_sents:]

        return doc_sent_encodings

    def _get_ids_and_labels_from_words(
            self,
            words: Sequence[Sequence[int]]
    ) -> Tuple[List[int], List[int]]:
        labels = []
        token_ids = []

        for word in words:
            if np.random.rand() < self._masked_lm_prob:
                labels.extend(word)

                if np.random.rand() < 0.8:
                    word = [self._mask_token_id] * len(word)
                elif np.random.rand() < 0.5:
                    word = np.random.randint(
                        low=0,
                        high=self._tokenizer.get_vocab_size(),
                        size=len(word)
                    )
            else:
                labels.extend([-100] * len(word))

            token_ids.extend(word)

        return token_ids, labels


def _get_words_from_sequences(sequences: Sequence[Encoding]) -> List[List[int]]:
    words = []
    for seq_a in sequences:
        word = []
        for token, id_ in zip(seq_a.tokens, seq_a.ids):
            if token.startswith('##') or not word:
                word.append(id_)
            else:
                words.append(word)
                word = []

        if word:
            words.append(word)

    return words


if __name__ == '__main__':
    tokenizer = BertWordPieceTokenizer(
        '/media/akarnachev/de415b0e-eedb-4ef5-bbae-ccb7a2495f33/rubert_cased_L-12_H-768_A-12_v1/vocab.txt')

    queue = Queue(maxsize=100000)

    producer = SamplesProducer(
        file_path=pathlib.Path('./train.dialogs'),
        tokenizer=tokenizer,
        tokenization_chunk_size=100,
        max_len=128,
        masked_lm_prob=0.15,
        out_samples_queue=queue
    )

    producer.start()

    while True:
        #sample = queue.get()
        print(queue.qsize())
        # print(tokenizer.decode(sample.token_ids, skip_special_tokens=False))

        #print(sample)
