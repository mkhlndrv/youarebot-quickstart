from pydantic import UUID4, BaseModel, StrictStr


class IncomingMessage(BaseModel):
    """A single message to classify."""

    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    """Classification result for one message."""

    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float
