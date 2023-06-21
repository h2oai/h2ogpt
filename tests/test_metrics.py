from tests.utils import wrap_test_forked


@wrap_test_forked
def test_bleurt():
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    import evaluate
    bleurt = evaluate.load("bleurt")
    results = bleurt.compute(predictions=predictions, references=references)
    assert [round(v, 2) for v in results["scores"]] == [1.03, 1.04]


@wrap_test_forked
def test_sacrebleu():
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
    import evaluate
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)

    assert list(results.keys()) == ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
    assert round(results["score"], 1) == 100.0

    predictions = ["hello there general kenobi", "on our way to ankh morpork"]
    references = [["hello there general kenobi", "hello there !"], ["goodbye ankh morpork", "ankh morpork"]]
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)
    assert list(results.keys()) == ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
    assert round(results["score"], 1) == 39.8


@wrap_test_forked
def test_bleu():
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [
        ["hello there general kenobi", "hello there!"],
        ["foo bar foobar"]
    ]
    import evaluate
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    assert results["bleu"] == 1.0


@wrap_test_forked
def test_squad_v1():
    predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
    references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    import evaluate
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    assert results == {'exact_match': 100.0, 'f1': 100.0}


def test_squad_v2():
    predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    import evaluate
    squad_v2_metric = evaluate.load("squad_v2")
    results = squad_v2_metric.compute(predictions=predictions, references=references)
    assert results == {'exact': 100.0, 'f1': 100.0, 'total': 1, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0,
                       'HasAns_total': 1, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0,
                       'best_f1_thresh': 0.0}


@wrap_test_forked
def test_rougue():
    import evaluate
    rouge = evaluate.load('rouge')
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    results = rouge.compute(predictions=predictions, references=references)
    assert results == {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}


@wrap_test_forked
def test_bertscore():
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    import evaluate
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    assert [round(v, 2) for v in results["f1"]] == [1.0, 1.0]


@wrap_test_forked
def test_chrf():
    prediction = ["The relationship between cats and dogs is not exactly friendly.",
                  "a good bookshop is just a genteel black hole that knows how to read."]
    reference = [["The relationship between dogs and cats is not exactly friendly.", ],
                 ["A good bookshop is just a genteel Black Hole that knows how to read."]]
    import evaluate
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=prediction, references=reference)
    assert results == {'score': 84.64214891738334, 'char_order': 6, 'word_order': 0, 'beta': 2}


@wrap_test_forked
def test_chrfpp():
    prediction = ["The relationship between cats and dogs is not exactly friendly.",
                  "a good bookshop is just a genteel black hole that knows how to read."]
    reference = [["The relationship between dogs and cats is not exactly friendly.", ],
                 ["A good bookshop is just a genteel Black Hole that knows how to read."]]
    import evaluate
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=prediction, references=reference, word_order=2)
    assert results == {'beta': 2, 'char_order': 6, 'score': 82.87263732906315, 'word_order': 2}


@wrap_test_forked
def test_wiki_split():
    sources = ["About 95 species are currently accepted ."]
    predictions = ["About 95 you now get in ."]
    references = [["About 95 species are currently known ."]]
    import evaluate
    wiki_split = evaluate.load("wiki_split")
    results = wiki_split.compute(sources=sources, predictions=predictions, references=references)
    assert results == {'sari': 21.805555555555557, 'sacrebleu': 14.535768424205482, 'exact': 0.0}


@wrap_test_forked
def test_super_glue():
    from evaluate import load
    # https://huggingface.co/datasets/boolq
    # passage, question, answer (as bool only though, but can ask LLM to only say true or false)
    super_glue_metric = load('super_glue', 'boolq')  # any of ["copa", "rte", "wic", "wsc", "wsc.fixed", "boolq", "axg"]
    predictions = [0, 1]
    references = [0, 1]
    results = super_glue_metric.compute(predictions=predictions, references=references)
    assert results == {'accuracy': 1.0}


@wrap_test_forked
def test_quip():
    from metrics.quip import Quip
    quip = Quip()

    predictions = ["Kathy's hair is green according to the first passage."]
    references = [["Kathy's hair is green.", "Bob is eating a sandwich.", "The sky is red with polka dots.",
                   "Alice went to the county fair.", "George is reading a newspaper."]]
    results = quip.compute(predictions=predictions, references=references)
    print(results)
    assert results == 0.16666666666666663

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [["Kathy's hair is green.", "Bob is eating a sandwich.", "The sky is red with polka dots.",
                   "Alice went to the county fair.", "George is reading a newspaper."]]
    results = quip.compute(predictions=predictions, references=references)
    print(results)
    assert results == 0.0

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [["chuck", "wood"]]
    results = quip.compute(predictions=predictions, references=references)
    print(results)
    assert results == 0.0

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [["chuck", "woodchuck"]]
    results = quip.compute(predictions=predictions, references=references)
    print(results)
    assert results == 0.0

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [["chuck", "woodchuck"]]
    results = quip.compute(predictions=predictions, references=references, min_len=1)
    print(results)
    assert results == 0.09523809523809523

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [["woodchuck chuck", "chuck"]]
    results = quip.compute(predictions=predictions, references=references)
    print(results)
    assert results == 0.05882352941176472

    predictions = ["The current goodwill balance is $25,173 million as of December 31, 2022."]
    references = [[
                      "Table 7.3: Goodwill (in millions) Consumer Banking and Lending Commercial Banking Corporate and Investment Banking Wealth and Investment Management Corporate Consolidated Company December 31, 2020 $ 16,418 3,018 5,375 1,276 305 26,392 Foreign currency translation — — — — — — Transfers of goodwill — (80) — (932) 1,012 — Divestitures — — — — (1,212) (1,212) December 31, 2021 $ 16,418 2,938 5,375 344 105 25,180 Foreign currency translation — (7) — — — (7) December 31, 2022 $ 16,418 2,931 5,375 344 105 25,173 Table 7.4 presents the components of other assets."]]
    results = quip.compute(predictions=predictions, references=references, min_len=1)
    print(results)
    assert results == 0.33333333333333337

    predictions = ["The current goodwill balance is $25,173 million as of December 31, 2022."]
    references = [[
                      "Table 7.3: Goodwill (in millions) Consumer Banking and Lending Commercial Banking Corporate and Investment Banking Wealth and Investment Management Corporate Consolidated Company December 31, 2020 $ 16,418 3,018 5,375 1,276 305 26,392 Foreign currency translation — — — — — — Transfers of goodwill — (80) — (932) 1,012 — Divestitures — — — — (1,212) (1,212) December 31, 2021 $ 16,418 2,938 5,375 344 105 25,180 Foreign currency translation — (7) — — — (7) December 31, 2022 $ 16,418 2,931 5,375 344 105 25,173 Table 7.4 presents the components of other assets."]]
    results = quip.compute(predictions=predictions, references=references, return_match_count=True)
    print(results)
    assert results == 4

    predictions = ["The current goodwill balance is $25,173 million as of December 31, 2022."]
    references = [[
                      "Table 7.3: Goodwill (in millions) Consumer Banking and Lending Commercial Banking Corporate and Investment Banking Wealth and Investment Management Corporate Consolidated Company December 31, 2020 $ 16,418 3,018 5,375 1,276 305 26,392 Foreign currency translation — — — — — — Transfers of goodwill — (80) — (932) 1,012 — Divestitures — — — — (1,212) (1,212) December 31, 2021 $ 16,418 2,938 5,375 344 105 25,180 Foreign currency translation — (7) — — — (7) December 31, 2022 $ 16,418 2,931 5,375 344 105 25,173 Table 7.4 presents the components of other assets."]]
    results = quip.compute(predictions=predictions, references=references, return_match_fraction_by_pred_length=True)
    print(results)
    assert results == 0.5

    predictions = ["How much wood would a woodchuck chuck if a woodchuck could chuck wood?"]
    references = [[
                      "Table 7.3: Goodwill (in millions) Consumer Banking and Lending Commercial Banking Corporate and Investment Banking Wealth and Investment Management Corporate Consolidated Company December 31, 2020 $ 16,418 3,018 5,375 1,276 305 26,392 Foreign currency translation — — — — — — Transfers of goodwill — (80) — (932) 1,012 — Divestitures — — — — (1,212) (1,212) December 31, 2021 $ 16,418 2,938 5,375 344 105 25,180 Foreign currency translation — (7) — — — (7) December 31, 2022 $ 16,418 2,931 5,375 344 105 25,173 Table 7.4 presents the components of other assets."]]
    results = quip.compute(predictions=predictions, references=references, return_match_fraction_by_pred_length=True)
    print(results)
    assert results == 0.0


@wrap_test_forked
def test_glue():
    # entailment
    """
    E.g. for qnli:
    The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs,
    where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding
    question (written by an annotator). The authors of the benchmark convert the task into sentence pair
    classification by forming a pair between each question and each sentence in the corresponding context,
    and filtering out pairs with low lexical overlap between the question and the context sentence.

    The task is to determine whether the context sentence contains the answer to the question.
    This modified version of the original task removes the requirement that the model select the exact answer,
    but also removes the simplifying assumptions that the answer is always present in the input
    and that lexical overlap is a reliable cue.
    :return:
    """
    from evaluate import load
    glue_metric = load('glue', 'qnli')
    references = [0, 1]
    predictions = [1, 1]
    results = glue_metric.compute(predictions=predictions, references=references)
    assert results == {'accuracy': 0.5}


@wrap_test_forked
def test_google_bleu():
    sentence1 = "the cat sat on the mat"
    sentence2 = "the cat ate the mat"
    import evaluate
    google_bleu = evaluate.load("google_bleu")
    result = google_bleu.compute(predictions=[sentence1], references=[[sentence2]])
    assert result == {'google_bleu': 0.3333333333333333}

    predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat',
                   'he read the book because he was interested in world history']
    references = [
        ['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat',
         'It is a guide to action that ensures that the rubber duck will never heed the cat commands',
         'It is the practical guide for the rubber duck army never to heed the directions of the cat'],
        ['he was interested in world history because he read the book']]
    google_bleu = evaluate.load("google_bleu")
    results = google_bleu.compute(predictions=predictions, references=references, min_len=2, max_len=6)
    assert round(results["google_bleu"], 2) == 0.4


@wrap_test_forked
def test_meteor():
    import evaluate
    meteor = evaluate.load('meteor')
    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands',
                   'It is the guiding principle which guarantees the military forces always being under the command of the Party',
                   'It is the practical guide for the army always to heed the directions of the party']]
    results = meteor.compute(predictions=predictions, references=references)
    assert round(results['meteor'], 2) == 0.69

    predictions = ["Kathy's hair is green according to the first passage."]
    references = [["Kathy's hair is green.", "Bob is eating a sandwich.", "The sky is red with polka dots.",
                   "Alice went to the county fair.", "George is reading a newspaper."]]
    results = meteor.compute(predictions=predictions, references=references)
    assert results == {'meteor': 0.9059829059829061}
    print(results)
