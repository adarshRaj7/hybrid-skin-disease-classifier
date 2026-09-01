import pytest

from skin_disease.inference import InvalidImageError, SkinDiseasePredictor


@pytest.fixture()
def predictor(tiny_checkpoint) -> SkinDiseasePredictor:
    return SkinDiseasePredictor(checkpoint_path=tiny_checkpoint)


def test_valid_image_produces_prediction(predictor, sample_rgb_image):
    result = predictor.predict(sample_rgb_image)
    assert "predicted_class" in result
    assert result["predicted_class"] in predictor.class_names


def test_top_3_predictions_returned(predictor, sample_rgb_image):
    result = predictor.predict(sample_rgb_image)
    assert len(result["top_3_predictions"]) == min(3, len(predictor.class_names))
    names = [p["class"] for p in result["top_3_predictions"]]
    assert len(names) == len(set(names))  # no duplicate classes


def test_confidence_values_are_valid_probabilities(predictor, sample_rgb_image):
    result = predictor.predict(sample_rgb_image)
    assert 0.0 <= result["confidence"] <= 1.0
    for pred in result["top_3_predictions"]:
        assert 0.0 <= pred["confidence"] <= 1.0
    # top_3 confidences must be sorted descending
    confidences = [p["confidence"] for p in result["top_3_predictions"]]
    assert confidences == sorted(confidences, reverse=True)


def test_class_mapping_is_correct(predictor, sample_rgb_image):
    result = predictor.predict(sample_rgb_image)
    assert result["predicted_class"] in predictor.class_names
    assert "severity" in result
    assert result["severity"] in {"MILD", "MODERATE", "SEVERE", "OTHER"}


def test_severe_class_maps_to_severe_severity(predictor):
    # "SkinCancer" is one of the fixture classes and is SEVERE per labels.py
    assert "SkinCancer" in predictor.class_names
    from skin_disease.labels import get_severity

    assert get_severity("SkinCancer") == "SEVERE"


def test_grayscale_and_rgba_images_are_handled(predictor, sample_grayscale_image, sample_rgba_image):
    result_gray = predictor.predict(sample_grayscale_image)
    result_rgba = predictor.predict(sample_rgba_image)
    assert result_gray["predicted_class"] in predictor.class_names
    assert result_rgba["predicted_class"] in predictor.class_names


def test_predict_from_raw_bytes(predictor, sample_image_bytes):
    result = predictor.predict(sample_image_bytes)
    assert result["predicted_class"] in predictor.class_names


def test_invalid_image_bytes_raise_gracefully(predictor):
    with pytest.raises(InvalidImageError):
        predictor.predict(b"this is not an image")


def test_missing_file_raises_gracefully(predictor):
    with pytest.raises(InvalidImageError):
        predictor.predict("/nonexistent/path/to/image.jpg")


def test_oversized_bytes_are_rejected(predictor, monkeypatch):
    import skin_disease.inference as inference_module

    monkeypatch.setattr(inference_module, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(InvalidImageError):
        predictor.predict(b"0123456789ABCDEF")  # 16 bytes > 10-byte limit
