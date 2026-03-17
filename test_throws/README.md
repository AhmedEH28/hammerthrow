# Test Throws Folder Structure

This directory should contain test throw folders, each representing a single hammer throw test.

## Required Folder Structure

Each test throw folder must contain:

1. **Two synchronized videos:**
   - `side_view.mp4` (or similar name containing "side")
   - `back_view.mp4` (or similar name containing "back")
   - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

2. **Camera calibration file:**
   - `calibration.csv` (or any `.csv`/`.txt` file not containing "metadata", "info", "release", or "frame" in the name)
   - Format: CSV with columns `X,Y,Z,x1,y1,x2,y2` (3D points and corresponding 2D points in both views)

3. **Metadata file (optional but recommended):**
   - `metadata.txt` or `info.json` (any file containing "metadata", "info", "release", or "frame" in the name)
   - Should contain release point and frame rate information

## Example Folder Structure:

```
test_throws/
├── throw_001/
│   ├── side_view.mp4
│   ├── back_view.mp4
│   ├── calibration.csv
│   └── metadata.txt
├── throw_002/
│   ├── side_camera.mp4
│   ├── back_camera.mp4
│   ├── camera_calibration.csv
│   └── throw_info.json
└── athlete_john_throw_3/
    ├── side.avi
    ├── back.avi
    ├── calibration_data.csv
    └── release_info.txt
```

## Metadata File Formats

### Option 1: JSON format
```json
{
    "release_point": 45,
    "frame_rate": 240.0,
    "athlete": "John Doe",
    "date": "2024-08-06"
}
```

### Option 2: Key-Value format
```
Release Point: 45
Frame Rate: 240
Athlete: John Doe
Date: 2024-08-06
```

## Notes:

- If no release point is specified, the system will analyze 30 frames before the last 50 frames
- If no frame rate is specified, the system will use the video's frame rate or default to 240 FPS
- Video file names should contain "side" or "back" to help with automatic identification
- Calibration files should NOT contain metadata keywords in their names

## Usage:

1. Place your test throw folders in this directory
2. Use the simplified analysis interface at `/simple`
3. Select the test throw and model
4. The system will automatically:
   - Find and process the videos
   - Load calibration data
   - Read metadata (release point, frame rate)
   - Calculate the optimal frame range (30 frames before release point, or 30 frames before last 50 frames if no release point)
   - Perform complete analysis and return the estimated throw distance
