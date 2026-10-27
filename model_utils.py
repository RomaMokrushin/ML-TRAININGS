def binary_postprocess(predictions):
    return [int(i >= 0.5) for i in predictions]
