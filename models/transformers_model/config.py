from pydantic import BaseModel

class TrainingParameters(BaseModel):
    num_train_epochs: int = 5
    learning_rate: float = 1e-05
    batch_size: int = 8
    weight_decay: int = 0.01
    warmup_steps: int = 500
    label_all_tokens: bool = True
    max_sequence_length = 512
    padding = "max_length"
    truncation = True
    gradient_accumulation_steps: int = 1
    