from src.pipeline import Pipeline

creation_matrix = [
    (64, "relu"),
    (256,"relu"),
    (512,"relu"),
    (1014,"relu")
]

compile_matrix = [
    {
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    }
]

pipe = Pipeline()

pipe.add_model(creation_matrix)
pipe.run_pipeline(compile_matrix)