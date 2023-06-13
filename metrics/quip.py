import datasets
import sacrebleu as scb
from packaging import version
from sacrebleu import CHRF

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
    'score' (float): The chrF (chrF++) score,
Examples:
    Example 1--a simple example of calculating chrF:
        >>> prediction = ["The relationship between cats and dogs is not exactly friendly.", "a good bookshop is just a genteel black hole that knows how to read."]
        >>> reference = [["The relationship between dogs and cats is not exactly friendly."], ["A good bookshop is just a genteel Black Hole that knows how to read."]]
        >>> chrf = evaluate.load("chrf")
        >>> results = chrf.compute(predictions=prediction, references=reference)
        >>> print(results)
        {'score': 84.64214891738334, 'char_order': 6, 'word_order': 0, 'beta': 2}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Quip(evaluate.Metric):
    def _info(self):
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, and the current version of `sacrebleu` doesn't match this condition.\n"
                'You can install it with `pip install "sacrebleu>=1.4.12"`.'
            )
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mjpost/sacreBLEU#chrf--chrf",
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
            codebase_urls=["https://github.com/mjpost/sacreBLEU#chrf--chrf"],
            reference_urls=[
                "https://github.com/m-popovic/chrF",
            ],
        )

    def _compute(
            self,
            predictions=None,
            references=None,
            char_order: int = CHRF.CHAR_ORDER,
            word_order: int = CHRF.WORD_ORDER,
            beta: int = CHRF.BETA,
            lowercase: bool = False,
            whitespace: bool = False,
            eps_smoothing: bool = False,
    ):
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]
        references_per_prediction = len(references[0])
        if any(len(refs) != references_per_prediction for refs in references):
            raise ValueError(
                "ChrF, as implemented by sacrebleu, requires the same number of references for each prediction"
            )
        transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]

        sb_chrf = CHRF(char_order, word_order, beta, lowercase, whitespace, eps_smoothing)
        output = sb_chrf.corpus_score(predictions, transformed_references)

        return {
            "score": output.score,
            "char_order": output.char_order,
            "word_order": output.word_order,
            "beta": output.beta,
        }

    @staticmethod
    def test1():
        prediction = ["The relationship between cats and dogs is not exactly friendly.",
                      "a good bookshop is just a genteel black hole that knows how to read."]
        reference = [["The relationship between dogs and cats is not exactly friendly.", ],
                     ["A good bookshop is just a genteel Black Hole that knows how to read."]]

        results = Quip.compute(predictions=prediction, references=reference)
        return results
        # {'score': 84.64214891738334, 'char_order': 6, 'word_order': 0, 'beta': 2}


if __name__ == '__main__':
    print(ChrF.test1())
