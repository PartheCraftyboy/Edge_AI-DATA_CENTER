"""Teaching the classifier from the reader.

A fake teacher stands in for the vision-language model throughout, so these
run without Ollama, a GPU or a second of latency. What they check is the part
that is easy to get wrong and expensive to discover later: that hand labels
are never replaced by machine ones, that the held-out splits stay honest, and
that the votes are cached rather than re-bought.
"""

import json

import cv2
import numpy as np
import pytest

from app import CLASS_NAMES
from app.data.real import PageStore
from app.distil import build_teacher_dataset, collect_votes, native_crop


class FakeTeacher:
    """Answers with a fixed class, and counts how often it was asked."""

    def __init__(self, answer="THEORY"):
        self.answer, self.calls = answer, 0

    def classify_region(self, image):
        self.calls += 1
        return self.answer, f"transcription {self.calls}"


def page_with_ink(h=600, w=800):
    page = np.full((h, w), 235, np.uint8)
    rng = np.random.default_rng(0)
    for y in range(80, 260, 26):
        for x in range(60, 560, 9):
            pts = np.array([[x + int(rng.integers(0, 6)), y - int(rng.integers(0, 14))]
                            for _ in range(4)], np.int32)
            cv2.polylines(page, [pts], False, 40, 2)
    return page


@pytest.fixture
def store(tmp_path):
    store = PageStore(root=tmp_path / "real")
    grey = page_with_ink()
    record = store.add(grey, [], source_name="page one")
    store.save_labels(record["id"], [{"box": [60, 70, 500, 200], "class": "THEORY"}])
    return store


class TestNativeCrop:
    def test_keeps_the_aspect_ratio_the_square_crop_throws_away(self):
        grey = np.full((400, 1200), 200, np.uint8)
        crop = native_crop(grey, [100, 100, 900, 120])
        assert crop.shape[1] > crop.shape[0] * 4, "a line must still read as long and short"

    def test_a_sliver_is_refused(self):
        assert native_crop(np.full((400, 400), 200, np.uint8), [0, 0, 4, 4]) is None


class TestCollectVotes:
    def test_a_vote_carries_its_transcription_as_evidence(self, store, tmp_path):
        out = collect_votes(store, reader=FakeTeacher("FORMULA"), cache=tmp_path / "v.json")
        vote = next(iter(out["votes"].values()))
        assert vote["class"] == "FORMULA"
        assert vote["transcription"], "a pseudo-label with no evidence cannot be audited"

    def test_votes_are_cached_not_bought_twice(self, store, tmp_path):
        cache = tmp_path / "v.json"
        first = FakeTeacher()
        collect_votes(store, reader=first, cache=cache)
        again = FakeTeacher()
        collect_votes(store, reader=again, cache=cache)
        assert first.calls > 0 and again.calls == 0

    def test_hand_label_is_recorded_beside_the_machine_one(self, store, tmp_path):
        out = collect_votes(store, reader=FakeTeacher("FORMULA"), cache=tmp_path / "v.json")
        assert any(v["hand"] == "THEORY" for v in out["votes"].values())

    def test_a_failing_teacher_does_not_end_the_pass(self, store, tmp_path):
        class Broken:
            def classify_region(self, image):
                raise RuntimeError("model fell over")

        out = collect_votes(store, reader=Broken(), cache=tmp_path / "v.json")
        assert out["failed"] >= 1 and out["votes"] == {}

    def test_limit_caps_the_number_of_calls(self, store, tmp_path):
        teacher = FakeTeacher()
        collect_votes(store, reader=teacher, cache=tmp_path / "v.json", limit=1)
        assert teacher.calls == 1


class TestReleaseTeacher:
    """The teacher holds 4196 MiB of a 6141 MiB card and Ollama's keep_alive
    is ten minutes, so a run that does not release it trains in the 1437 MiB
    left over for the first ten minutes of every round."""

    def test_the_teacher_is_released_after_the_votes_are_in(self, store, tmp_path):
        class Unloadable(FakeTeacher):
            unloaded = False

            def unload(self):
                self.unloaded = True

        teacher = Unloadable()
        out = collect_votes(store, reader=teacher, cache=tmp_path / "v.json")
        assert teacher.unloaded and out["freed_teacher"]

    def test_a_teacher_with_close_but_no_unload_is_still_released(self, store, tmp_path):
        from app.distil import release_teacher

        class Closeable:
            closed = False

            def close(self):
                self.closed = True

        teacher = Closeable()
        assert release_teacher(teacher) and teacher.closed

    def test_a_teacher_that_cannot_be_released_does_not_raise(self, store, tmp_path):
        from app.distil import release_teacher

        class Stuck:
            def unload(self):
                raise RuntimeError("ollama is gone")

        assert release_teacher(Stuck()) is False

    def test_a_teacher_with_no_release_at_all_is_tolerated(self):
        from app.distil import release_teacher

        assert release_teacher(object()) is False


class TestFailureRecovery:
    def test_a_failing_crop_resets_the_model(self, store, tmp_path):
        """A vision model can enter a state where every reply is empty. One
        oversized region did that and the next 414 regions failed against the
        same poisoned runner. Each failure must reset it."""
        class Flaky:
            def __init__(self):
                self.calls = 0
                self.unloads = 0

            def classify_region(self, image):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("empty reply")
                return "THEORY", "text"

            def unload(self):
                self.unloads += 1

        teacher = Flaky()
        collect_votes(store, reader=teacher, cache=tmp_path / "v.json")
        assert teacher.unloads >= 1, "a failure must reset the model"

    def test_failure_reasons_are_reported_not_just_counted(self, store, tmp_path):
        class Broken:
            def classify_region(self, image):
                raise RuntimeError("token repeat limit reached")

        out = collect_votes(store, reader=Broken(), cache=tmp_path / "v.json")
        assert out["failed"] >= 1
        assert any("token repeat limit" in r for r in out["reasons"]), \
            "counting failures without their cause is how 414 of them said nothing"


class TestCheckpointIsOverwritten:
    def test_training_twice_leaves_one_best_checkpoint(self, tmp_path):
        """Lightning adds a version counter when the file exists, so a second
        round wrote best-v1.ckpt while the caller kept reading best.ckpt and
        scored the first round's model twice."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        import inspect

        from app.train import train

        source = inspect.getsource(train)
        assert "enable_version_counter=False" in source
        assert "enable_version_counter" in inspect.signature(ModelCheckpoint.__init__).parameters


class TestEveryCorpusInOnePass:
    """Training one corpus per run threw the earlier corpora away: each run
    restarts from the synthetic checkpoint and writes the same best.ckpt, so
    GNHK then IAM shipped a model that had only seen IAM."""

    def test_several_folders_are_read_in_one_call(self, store, tmp_path):
        from app.distil import page_images

        a, b = tmp_path / "corpus_a", tmp_path / "corpus_b"
        for folder, name in ((a, "alpha"), (b, "beta")):
            folder.mkdir()
            cv2.imwrite(str(folder / f"{name}.png"), page_with_ink())

        found = {pid for pid, _, _, _, _ in page_images(store, [a, b])}
        assert "ext-alpha" in found and "ext-beta" in found

    def test_a_single_folder_still_works(self, store, tmp_path):
        from app.distil import page_images

        a = tmp_path / "corpus_a"
        a.mkdir()
        cv2.imwrite(str(a / "alpha.png"), page_with_ink())
        assert any(pid == "ext-alpha" for pid, _, _, _, _ in page_images(store, a))

    def test_one_dataset_holds_every_corpus(self, store, tmp_path):
        """The symptom to catch: a dataset built from the second corpus
        alone, with the first one's crops missing entirely."""
        a, b = tmp_path / "corpus_a", tmp_path / "corpus_b"
        for folder, name in ((a, "alpha"), (b, "beta")):
            folder.mkdir()
            cv2.imwrite(str(folder / f"{name}.png"), page_with_ink())

        votes = collect_votes(store, [a, b], reader=FakeTeacher(),
                              cache=tmp_path / "v.json")["votes"]
        pages = {v["page"] for v in votes.values()}
        assert "ext-alpha" in pages and "ext-beta" in pages

        manifest = build_teacher_dataset(votes, store, tmp_path / "ds", [a, b])
        assert manifest["counts"]["train"] > 0
        assert manifest["teacher_labels"] >= 2

    def test_as_dirs_accepts_one_or_many_or_none(self):
        from app.distil import as_dirs

        assert len(as_dirs("a")) == 1
        assert len(as_dirs(["a", "b"])) == 2
        assert as_dirs(None) == []


class TestCorrectionsCountMore:
    """A crop a person corrected is ground truth *and* a place the model was
    demonstrably wrong, which makes it the most informative thing in the
    set. A confirmation is truth the model already agrees with."""

    def test_a_correction_outweighs_a_confirmation_outweighs_a_guess(self):
        from app.distil import copies_for

        guess = copies_for({"hand": None, "class": "THEORY"}, 2)
        confirmed = copies_for({"hand": "THEORY", "class": "THEORY"}, 2)
        corrected = copies_for({"hand": "THEORY", "class": "FORMULA"}, 2)
        assert corrected > confirmed > guess

    def test_a_hand_label_is_a_correction_when_it_differs_from_the_vote(self):
        """No bookkeeping is needed to tell them apart: disagreeing with the
        teacher is what makes a hand label a correction."""
        from app.distil import CORRECTED_COPIES, CONFIRMED_COPIES, copies_for

        assert copies_for({"hand": "TABLE", "class": "FORMULA"}, 2) == CORRECTED_COPIES
        assert copies_for({"hand": "TABLE", "class": "TABLE"}, 2) == CONFIRMED_COPIES

    def test_the_manifest_counts_corrections_separately(self, store, tmp_path):
        votes = collect_votes(store, reader=FakeTeacher("FORMULA"),
                              cache=tmp_path / "v.json")["votes"]
        manifest = build_teacher_dataset(votes, store, tmp_path / "ds")
        # The fixture's hand label is THEORY and the fake teacher says
        # FORMULA, so every hand label here is a correction.
        assert manifest["hand_corrected"] >= 1
        assert manifest["hand_confirmed"] == 0

    def test_a_corrected_crop_really_appears_more_often(self, store, tmp_path):
        """Same regions, same split, counted twice: once as corrections and
        once as guesses. The split is pinned because stripping `hand` also
        changes which split a region is allowed into, and an earlier version
        of this test measured that instead and failed about half the time.
        """
        votes = collect_votes(store, reader=FakeTeacher("FORMULA"),
                              cache=tmp_path / "v.json")["votes"]
        assert votes, "the fixture page produced no regions"

        corrected = {k: {**v, "split": "train"} for k, v in votes.items()}
        guessed = {k: {**v, "split": "train", "hand": None} for k, v in votes.items()}
        assert any(v["hand"] and v["hand"] != v["class"] for v in corrected.values())

        build_teacher_dataset(corrected, store, tmp_path / "a", jitter_copies=2)
        build_teacher_dataset(guessed, store, tmp_path / "b", jitter_copies=2)
        a = len(np.load(tmp_path / "a" / "crops" / "train_labels.npy"))
        b = len(np.load(tmp_path / "b" / "crops" / "train_labels.npy"))
        assert a > b, f"a correction should weigh more than a guess ({a} vs {b})"


class TestBuildDataset:
    def test_hand_labels_beat_the_teacher(self, store, tmp_path):
        """The teacher says FORMULA, a person said THEORY. The person wins, or
        a labelling session is undone by the next distillation round."""
        votes = collect_votes(store, reader=FakeTeacher("FORMULA"),
                              cache=tmp_path / "v.json")["votes"]
        build_teacher_dataset(votes, store, tmp_path / "ds")
        labels = np.concatenate([
            np.load(tmp_path / "ds" / "crops" / f"{s}_labels.npy")
            for s in ("train", "val", "test")])
        assert len(labels) and set(labels.tolist()) == {CLASS_NAMES.index("THEORY")}

    def test_val_and_test_keep_only_hand_labelled_regions(self, store, tmp_path):
        """Scoring against the teacher only proves the student copied it."""
        grey = page_with_ink()
        unlabelled = store.add(grey, [], source_name="never labelled")
        votes = collect_votes(store, reader=FakeTeacher(), cache=tmp_path / "v.json")["votes"]
        for vote in votes.values():
            vote["split"] = "val"
        manifest = build_teacher_dataset(votes, store, tmp_path / "ds")
        hand = sum(1 for v in votes.values() if v["hand"])
        assert manifest["counts"]["val"] == hand
        assert unlabelled["id"] in {v["page"] for v in votes.values()}

    def test_manifest_separates_taught_from_hand_counts(self, store, tmp_path):
        votes = collect_votes(store, reader=FakeTeacher(), cache=tmp_path / "v.json")["votes"]
        manifest = build_teacher_dataset(votes, store, tmp_path / "ds")
        assert manifest["hand_labels"] >= 1
        assert set(manifest["per_class"]) == set(CLASS_NAMES)

    def test_crops_match_the_training_recipe(self, store, tmp_path):
        from app.data.real import CROP_SIZE

        votes = collect_votes(store, reader=FakeTeacher(), cache=tmp_path / "v.json")["votes"]
        build_teacher_dataset(votes, store, tmp_path / "ds")
        crops = np.load(tmp_path / "ds" / "crops" / "train_crops.npy")
        assert crops.ndim == 3 and crops.shape[1:] == (CROP_SIZE, CROP_SIZE)
        assert crops.dtype == np.uint8


class TestDisagreementPenalty:
    def test_penalty_weights_the_crops_the_student_gets_wrong(self):
        """The loss is a weighted mean, so it exceeds plain cross-entropy
        only when the disagreeing crops are the costlier ones. An earlier
        version of this test used random logits against random labels,
        where every per-sample loss sits at ln(5) and the comparison is a
        coin flip; it passed most runs and failed some. The logits here are
        constructed so the disagreements are unambiguously the expensive
        ones."""
        import torch

        from app.models.classifier import RegionClassifier

        plain = RegionClassifier(backbone="squeezenet1_1", pretrained=False)
        penalised = RegionClassifier(backbone="squeezenet1_1", pretrained=False,
                                     disagreement_penalty=5.0)
        y = torch.tensor([0, 0, 1, 1])
        # Two confident and right, two confident and wrong.
        logits = torch.tensor([
            [9.0, 0.0, 0.0, 0.0, 0.0],   # says 0, is 0  -> agrees, cheap
            [9.0, 0.0, 0.0, 0.0, 0.0],   # says 0, is 0  -> agrees, cheap
            [9.0, 0.0, 0.0, 0.0, 0.0],   # says 0, is 1  -> disagrees, dear
            [9.0, 0.0, 0.0, 0.0, 0.0],   # says 0, is 1  -> disagrees, dear
        ])
        per_sample = plain.per_sample_loss(logits, y)
        disagrees = logits.argmax(dim=1) != y
        weights = torch.where(disagrees, 5.0, 1.0)

        flat = per_sample.mean()
        weighted = (weights * per_sample).sum() / weights.sum()
        assert weighted > flat, "disagreements must cost more than agreements"
        assert penalised.hparams.disagreement_penalty == 5.0

    def test_with_full_agreement_the_penalty_vanishes(self):
        """Every crop at weight one is ordinary cross-entropy, which is what
        makes the objective anneal rather than needing a stopping rule."""
        import torch

        from app.models.classifier import RegionClassifier

        model = RegionClassifier(backbone="squeezenet1_1", pretrained=False)
        y = torch.tensor([0, 0])
        logits = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0],
                               [9.0, 0.0, 0.0, 0.0, 0.0]])
        per_sample = model.per_sample_loss(logits, y)
        weights = torch.where(logits.argmax(dim=1) != y, 5.0, 1.0)
        assert torch.isclose((weights * per_sample).sum() / weights.sum(),
                             per_sample.mean())

    def test_default_is_no_penalty(self):
        from app.models.classifier import RegionClassifier

        assert RegionClassifier(backbone="squeezenet1_1",
                                pretrained=False).hparams.disagreement_penalty == 1.0
