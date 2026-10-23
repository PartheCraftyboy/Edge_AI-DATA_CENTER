"""Dataset and split tests.

Two failure modes here are both silent and both fatal to the reported numbers:

**Leakage across the split.** Crops are cut from pages, and several crops come
from the same page, sharing its paper texture, its ruling, its ink darkness and
its camera warp. Splitting by crop would put siblings on both sides of the
boundary and inflate validation accuracy by a margin nobody could explain. The
split is therefore by page, and that is asserted rather than assumed.

**Augmenting the evaluation splits.** Validation and test must see clean crops,
or the number being reported is accuracy under a random perturbation that
changes every run.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import CLASS_NAMES, NUM_CLASSES
from app.data.dataset import (
    SPLIT_FRACTIONS,
    CropDataset,
    PalimpsestDataModule,
    build_dataset,
    split_for_seed,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A small real dataset. Small enough to build per test session, real
    enough that nothing here is mocked."""
    out = tmp_path_factory.mktemp("ds")
    return build_dataset(out, n_pages=36, save_pages=4, warp_severity=0.6, seed=0)


class TestSplitDiscipline:
    def test_fractions_sum_to_one(self):
        assert abs(sum(SPLIT_FRACTIONS.values()) - 1.0) < 1e-9

    def test_a_seed_maps_to_exactly_one_split(self):
        for seed in range(500):
            splits = [s for s in SPLIT_FRACTIONS if split_for_seed(seed, 500) == s]
            assert len(splits) == 1

    def test_split_sizes_are_close_to_the_fractions(self):
        counts = {s: 0 for s in SPLIT_FRACTIONS}
        for seed in range(1000):
            counts[split_for_seed(seed, 1000)] += 1
        for name, want in SPLIT_FRACTIONS.items():
            assert abs(counts[name] / 1000 - want) < 0.02, counts

    def test_no_page_appears_in_two_splits(self, built):
        seen = {}
        for name in SPLIT_FRACTIONS:
            for seed in built["page_seeds"][name]:
                assert seed not in seen, f"page {seed} in {seen.get(seed)} and {name}"
                seen[seed] = name

    def test_every_split_is_non_empty(self, built):
        for name in SPLIT_FRACTIONS:
            assert built["counts"][name] > 0, built["counts"]


class TestCrops:
    def test_shape_and_dtype_on_disk(self, built):
        crops = np.load(built["paths"]["train_crops"], mmap_mode="r")
        assert crops.ndim == 3 and crops.shape[1:] == (224, 224)
        assert crops.dtype == np.uint8

    def test_labels_are_valid_class_indices(self, built):
        for name in SPLIT_FRACTIONS:
            labels = np.load(built["paths"][f"{name}_labels"])
            assert labels.dtype == np.int64
            assert int(labels.min()) >= 0 and int(labels.max()) < NUM_CLASSES

    def test_every_class_is_present_in_training(self, built):
        labels = np.load(built["paths"]["train_labels"])
        assert set(np.unique(labels).tolist()) == set(range(NUM_CLASSES)), (
            "a class is missing; the confusion matrix would have an empty row"
        )

    def test_classes_are_roughly_balanced(self, built):
        labels = np.load(built["paths"]["train_labels"])
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        assert counts.min() / counts.max() > 0.5, dict(zip(CLASS_NAMES, counts.tolist()))


class TestDatasetTensors:
    def test_returns_three_channel_float_tensors(self, built):
        ds = CropDataset(built["paths"]["val_crops"], built["paths"]["val_labels"], augment=False)
        x, y = ds[0]
        assert x.shape == (3, 224, 224)
        assert x.dtype == torch.float32
        assert isinstance(y, int) or y.dtype == torch.int64

    def test_normalisation_centres_the_data(self, built):
        """ImageNet statistics, because the backbones are ImageNet-initialised
        and feeding them raw 0-1 pixels wastes the pretraining."""
        ds = CropDataset(built["paths"]["val_crops"], built["paths"]["val_labels"], augment=False)
        x, _ = ds[0]
        assert -4.0 < float(x.mean()) < 4.0
        assert float(x.min()) < 0.0, "data does not look normalised"

    def test_evaluation_split_is_deterministic(self, built):
        ds = CropDataset(built["paths"]["val_crops"], built["paths"]["val_labels"], augment=False)
        assert torch.equal(ds[3][0], ds[3][0])

    def test_training_split_varies_between_epochs(self, built):
        """Augmentation must actually fire, or Module 4 contributes nothing."""
        ds = CropDataset(built["paths"]["train_crops"], built["paths"]["train_labels"],
                         augment=True, seed=0)
        ds.set_epoch(0)
        first = ds[2][0].clone()
        ds.set_epoch(1)
        assert not torch.equal(first, ds[2][0])

    def test_augmentation_is_reproducible_for_a_given_epoch(self, built):
        a = CropDataset(built["paths"]["train_crops"], built["paths"]["train_labels"],
                        augment=True, seed=7)
        b = CropDataset(built["paths"]["train_crops"], built["paths"]["train_labels"],
                        augment=True, seed=7)
        a.set_epoch(3), b.set_epoch(3)
        assert torch.equal(a[5][0], b[5][0])

    def test_labels_survive_augmentation(self, built):
        ds = CropDataset(built["paths"]["train_crops"], built["paths"]["train_labels"],
                         augment=True, seed=0)
        raw = np.load(built["paths"]["train_labels"])
        for i in (0, 4, 9):
            assert ds[i][1] == int(raw[i])


class TestDataModule:
    def test_batches_have_the_expected_shape(self, built):
        dm = PalimpsestDataModule(built["root"], batch_size=4, num_workers=0)
        dm.setup("fit")
        x, y = next(iter(dm.train_dataloader()))
        assert x.shape == (4, 3, 224, 224)
        assert y.shape == (4,)

    def test_validation_loader_does_not_augment(self, built):
        dm = PalimpsestDataModule(built["root"], batch_size=4, num_workers=0)
        dm.setup("fit")
        assert dm.val_set.augment is False

    def test_test_split_is_available_separately(self, built):
        dm = PalimpsestDataModule(built["root"], batch_size=4, num_workers=0)
        dm.setup("test")
        x, _ = next(iter(dm.test_dataloader()))
        assert x.shape[1:] == (3, 224, 224)
