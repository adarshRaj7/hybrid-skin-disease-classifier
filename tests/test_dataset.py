from skin_disease.dataset import SkinLesionDataset, make_stratified_split, validate_directory
from skin_disease.labels import ClassLabels


def test_class_labels_from_directory_is_deterministic(synthetic_dataset_dir):
    labels_a = ClassLabels.from_directory(synthetic_dataset_dir / "train")
    labels_b = ClassLabels.from_directory(synthetic_dataset_dir / "train")
    assert labels_a.classes == labels_b.classes
    assert labels_a.classes == sorted(labels_a.classes)


def test_class_labels_json_round_trip(tmp_path, class_labels):
    path = tmp_path / "class_names.json"
    class_labels.save(path)
    loaded = ClassLabels.load(path)
    assert loaded.classes == class_labels.classes
    assert loaded.class_to_idx == class_labels.class_to_idx


def test_dataset_loads_all_synthetic_images(synthetic_dataset_dir, class_labels):
    dataset = SkinLesionDataset(synthetic_dataset_dir / "train", class_labels)
    assert len(dataset) == 6 * class_labels.num_classes
    image, label = dataset[0]
    assert image.size == (48, 48)
    assert 0 <= label < class_labels.num_classes


def test_stratified_split_is_disjoint_and_covers_all_indices():
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    train_idx, val_idx = make_stratified_split(labels, val_fraction=0.25, seed=0)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(len(labels)))


def test_stratified_split_preserves_class_proportions_roughly():
    labels = [0] * 20 + [1] * 20
    train_idx, val_idx = make_stratified_split(labels, val_fraction=0.5, seed=0)
    val_labels = [labels[i] for i in val_idx]
    assert val_labels.count(0) == val_labels.count(1)


def test_validate_directory_reports_unreadable_files(tmp_path):
    class_dir = tmp_path / "BadClass"
    class_dir.mkdir()
    (class_dir / "not_really_an_image.jpg").write_bytes(b"not an image")
    unreadable, total = validate_directory(tmp_path)
    assert total == 1
    assert len(unreadable) == 1
