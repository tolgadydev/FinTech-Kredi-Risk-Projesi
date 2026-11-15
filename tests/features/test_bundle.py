from credit_features.bundle import Bundle, load_bundle


def test_training_code_uses_the_shared_loader():
    from ml.train_final import Bundle as TrainingBundle
    from ml.train_final import load_bundle as training_loader

    assert training_loader is load_bundle
    assert TrainingBundle is Bundle