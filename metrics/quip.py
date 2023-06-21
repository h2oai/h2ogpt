import os

import datasets
import pandas as pd
import sacrebleu as scb
from packaging import version
from sacrebleu import CHRF
import string

import evaluate

_CITATION = """\
@ARTICLE{2023arXiv230513252W,
       author = {{Weller}, Orion and {Marone}, Marc and {Weir}, Nathaniel and {Lawrie}, Dawn and {Khashabi}, Daniel and {Van Durme}, Benjamin},
        title = "{``According to ...'' Prompting Language Models Improves Quoting from Pre-Training Data}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language, Computer Science - Artificial Intelligence},
         year = 2023,
        month = may,
          eid = {arXiv:2305.13252},
        pages = {arXiv:2305.13252},
          doi = {10.48550/arXiv.2305.13252},
archivePrefix = {arXiv},
       eprint = {2305.13252},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230513252W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""

_DESCRIPTION = """\
In order to understand whether models are able
to ground to their pre-training data, we first need
to have a way of measuring this phenomena. We
adopt a narrow definition of grounding (quoting
from source material) while acknowledging that
grounding is a broad term.
To enable fast and efficient measurement of
quoting from pre-training data for many language
model generations across large corpora, we build
off of a D ATA P ORTRAIT (Marone and Van Durme,
2023), which allows for fast membership queries
for each n-gram in the output. This approach en-
ables us to perform a one-time indexing of a large
corpus (e.g. Wikipedia) and at inference time sim-
ply compute a constant time lookup operation (in
milliseconds) for each n-gram in the generation.
We build a D ATA P ORTRAIT on the version of
Wikipedia included in the Pile, 2 as it allows for
us to exactly test the pre-training data included
in many models like GPT-J and is similar to the
training data used in T5. However, we note that for
some models evaluated in this paper (e.g. OpenAI
models) there is no public information about the
Wikipedia version in the models.
We use character based n-grams as opposed to a
token-based n-gram as different models have differ-
ent tokenization schemes; furthermore, character-
based n-gram metrics have widespread usage in
fields such as machine translation with metrics like
chrF and chrF++ (Popović, 2015, 2017). We use
25 character grams for the sketch, approximately 5-
gram words, as we found it empirically gave mean-
ingful results (not too small of an n-gram and not
too large). The D ATA P ORTRAIT checks for exact
matches and is sensitive to orthographic variation
(e.g. case, whitespace). Therefore we view this as
a lower-bound on actual quoting performance.
We define our new metric QUIP-Score as the
character n-gram precision of the generated out-
put compared to the pre-training corpus. More
formally, for generation Y and text corpus C:
P
gram n ∈Y 1 C (gram n )
QUIP(Y ; C) =
,
|gram n ∈ Y |
where 1(.) is an indicator function: 1 if gram n ∈ C
else 0. Thus, a score of 0.5 would indicate that
50% of the generated text n-grams are found in
the pre-training corpus. We macro-average this
quantity over a set of generations to obtain a single
performance number for a given test dataset. 3
"""

_KWARGS_DESCRIPTION = """
Produces QUIP scores for checking grounding from references
Args:
    predictions (list of str): The predicted sentences.
    references (list of list of str): The references. There should be one reference sub-list for each prediction sentence.
Returns:
    'score' (float): The QUIP score,
Examples:
    Example 1--a simple example of calculating chrF:
    predictions = ["The current goodwill balance is $25,173 million as of December 31, 2022."]
    references = [[
                      "Table 7.3: Goodwill (in millions) Consumer Banking and Lending Commercial Banking Corporate and Investment Banking Wealth and Investment Management Corporate Consolidated Company December 31, 2020 $ 16,418 3,018 5,375 1,276 305 26,392 Foreign currency translation — — — — — — Transfers of goodwill — (80) — (932) 1,012 — Divestitures — — — — (1,212) (1,212) December 31, 2021 $ 16,418 2,938 5,375 344 105 25,180 Foreign currency translation — (7) — — — (7) December 31, 2022 $ 16,418 2,931 5,375 344 105 25,173 Table 7.4 presents the components of other assets."]]
    results = quip.compute(predictions=predictions, references=references, return_match_fraction_by_pred_length=True)
    print(results)
    assert results == 0.5
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Quip(evaluate.Metric):
    def __init__(self, **kwargs):

        self.set_common = None
        if False:
            common_words_file = "data/NGSL_1.2_stats.csv.zip"
            if os.path.isfile(common_words_file):
                df = pd.read_csv(common_words_file)
                self.set_common = set(df['Lemma'].values.tolist())
        else:
            # https://norvig.com/ngrams/count_1w.txt
            common_words_file = "data/count_1w.txt.zip"
            if os.path.isfile(common_words_file):
                df = pd.read_csv(common_words_file, names=["word", "freq"], header=None, sep='\t')
                df = df.head(1000)
                self.set_common = set(df['word'].values.tolist())
                for k in list(string.ascii_lowercase):
                    keep = {'i', 'I', 'A', 'a'}
                    if k in self.set_common:
                        if k in keep:
                            continue
                        self.set_common.remove(k)

        super().__init__(**kwargs)

    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `quip`, the module `sacrebleu>=1.4.12` is required, and the current version of `sacrebleu` doesn't match this condition.\n"
                'You can install it with `pip install "sacrebleu>=1.4.12"`.'
            )
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/h2oai/h2ogpt",
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/h2oai/h2ogpt"],
            reference_urls=[
                "https://github.com/h2oai/h2ogpt",
            ],
        )

    def _compute(
            self,
            predictions=None,
            references=None,
            reduced=True,
            min_len=2,
            max_len=5,
            return_match_count=False,
            return_match_fraction_by_pred_length=False,
            **kwargs,
    ):
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError(
                "Quip requires the same number of references for each prediction"
            )
        # transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]

        if reduced:
            punc = """"!"#$%&()*+,-./:;<=>?@[\\]^_{|}~"""

            for predi, pred in enumerate(predictions):
                pred = pred.translate(str.maketrans(punc, ' ' * len(punc))).strip()
                predictions[predi] = ' '.join([x for x in pred.split() if x not in self.set_common])

            for refi, refl in enumerate(references):
                for refj, ref in enumerate(refl):
                    ref = ref.translate(str.maketrans(punc, ' ' * len(punc))).strip()
                    references[refi][refj] = ' '.join([x for x in ref.split() if x not in self.set_common])

        from nltk.util import everygrams
        from utils import flatten_list
        pred_ngrams = set(
            flatten_list([list(everygrams(x.split(), min_len=min_len, max_len=max_len)) for x in predictions]))
        ref_ngrams = set(flatten_list(
            [[list(everygrams(y.split(), min_len=min_len, max_len=max_len)) for y in z] for z in references]))
        residual = pred_ngrams.difference(ref_ngrams)
        if return_match_count:
            return len(pred_ngrams) - len(residual)
        else:
            if not return_match_fraction_by_pred_length:
                # Score = 0.0: No match
                # Score = 1.0: Perfect match
                return 1.0 - len(residual) / len(pred_ngrams)
            else:
                # FIXME: only works with 1 prediction
                nmatches = len(pred_ngrams) - len(residual)
                return min(1.0, nmatches / len(predictions[0].split()))

    def get_reduced_size(self, reduced_query, verbose=True):
        reduced_query_words = reduced_query.split(' ')
        set_common = set(self.df['Lemma'].values.tolist())
        num_common = len([x.lower() in set_common for x in reduced_query_words])
        frac_common = num_common / len(reduced_query) if reduced_query else 0
        # FIXME: report to user bad query that uses too many common words
        if verbose:
            print("frac_common: %s" % frac_common, flush=True)
