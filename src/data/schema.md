# Landmark Dataset Schema (Google - Isolated Sign Language Recognition)

## Columns
- frame: int
    - Frame number in the raw video.
- row_id: str
    - Unique identifier for the row (format: frame-type-landmark_index).
- type: str
    - Landmark type. One of ['face', 'left_hand', 'pose', 'right_hand'].
- landmark_index: int
    - Index of the landmark within its type.
- x: float
    - Normalized x coordinate (0 to 1).
- y: float
    - Normalized y coordinate (0 to 1).
- z: float
    - Normalized z coordinate (depth; may be unreliable).

## Notes
- Each .parquet file contains landmark data for a single gesture sequence.
- Not all frames have all landmark types present.
- The dataset is for gesture recognition, not identity recognition.
- Only x, y, z are used for model inference.