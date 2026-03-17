import os
import json
import glob
from typing import Dict, List, Tuple, Optional

class TestThrowManager:
    """
    Manages test throw folders and automatically extracts necessary files and metadata
    """
    
    def __init__(self, test_throws_directory: str):
        """
        Initialize with the base directory containing test throw folders
        
        Args:
            test_throws_directory: Path to directory containing test throw folders
        """
        self.test_throws_directory = test_throws_directory
        
    def get_available_test_throws(self) -> List[Dict]:
        """
        Scan the test throws directory and return available test throws
        
        Returns:
            List of dictionaries containing test throw information
        """
        test_throws = []
        
        if not os.path.exists(self.test_throws_directory):
            return test_throws
            
        # Get all subdirectories (each should be a test throw)
        for folder_name in os.listdir(self.test_throws_directory):
            folder_path = os.path.join(self.test_throws_directory, folder_name)
            
            if os.path.isdir(folder_path):
                throw_info = self._analyze_test_throw_folder(folder_path, folder_name)
                if throw_info:
                    test_throws.append(throw_info)
                    
        return sorted(test_throws, key=lambda x: x['name'])
    
    def _analyze_test_throw_folder(self, folder_path: str, folder_name: str) -> Optional[Dict]:
        """
        Analyze a test throw folder to extract file information
        
        Args:
            folder_path: Path to the test throw folder
            folder_name: Name of the folder
            
        Returns:
            Dictionary with test throw information or None if invalid
        """
        try:
            # Look for video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(folder_path, ext)))
                video_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            # Look for calibration files
            calib_extensions = ['*.csv', '*.txt', '*.json']
            calibration_files = []
            for ext in calib_extensions:
                calibration_files.extend(glob.glob(os.path.join(folder_path, ext)))
                calibration_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            # Filter out files that might be metadata (not calibration)
            metadata_files = []
            actual_calibration_files = []
            
            for calib_file in calibration_files:
                filename = os.path.basename(calib_file).lower()
                if any(keyword in filename for keyword in ['metadata', 'info', 'release', 'frame']) or filename.endswith('.json'):
                    metadata_files.append(calib_file)
                else:
                    actual_calibration_files.append(calib_file)
            
            # Try to identify side and back videos
            side_video = None
            back_video = None
            
            for video in video_files:
                filename = os.path.basename(video).lower()
                if 'side' in filename:
                    side_video = video
                elif 'back' in filename:
                    back_video = video
            
            # If we couldn't identify by name, just take first two videos
            if not side_video and not back_video and len(video_files) >= 2:
                side_video = video_files[0]
                back_video = video_files[1]
            
            # Must have at least 2 videos and 1 calibration file
            if len(video_files) < 2 or len(actual_calibration_files) < 1:
                return None
            
            # Look for metadata/info file
            metadata_info = self._extract_metadata_info(metadata_files)
            
            print(f"Debug: Analyzing folder {folder_name}")
            print(f"Debug: Found {len(video_files)} video files: {[os.path.basename(f) for f in video_files]}")
            print(f"Debug: Found {len(actual_calibration_files)} calibration files: {[os.path.basename(f) for f in actual_calibration_files]}")
            print(f"Debug: Found {len(metadata_files)} metadata files: {[os.path.basename(f) for f in metadata_files]}")
            print(f"Debug: Extracted metadata: {metadata_info}")
            
            return {
                'name': folder_name,
                'path': folder_path,
                'side_video': side_video,
                'back_video': back_video,
                'calibration_file': actual_calibration_files[0],  # Take first calibration file
                'metadata_file': metadata_files[0] if metadata_files else None,
                'release_point': metadata_info.get('release_point'),
                'frame_rate': metadata_info.get('frame_rate'),
                'valid': True
            }
            
        except Exception as e:
            print(f"Error analyzing folder {folder_name}: {e}")
            return None
    
    def _extract_metadata_info(self, metadata_files: List[str]) -> Dict:
        """
        Extract release point and frame rate from metadata files
        
        Args:
            metadata_files: List of potential metadata files
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata_info = {}
        
        for file_path in metadata_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if 'release_point' in data:
                        metadata_info['release_point'] = int(data['release_point'])
                    if 'frame_rate' in data:
                        metadata_info['frame_rate'] = float(data['frame_rate'])
                    continue
                except json.JSONDecodeError:
                    pass
                
                # Try to parse as key-value pairs
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        
                        if 'release' in key and 'point' in key:
                            try:
                                metadata_info['release_point'] = int(value)
                            except ValueError:
                                pass
                        elif 'frame' in key and 'rate' in key:
                            try:
                                metadata_info['frame_rate'] = float(value)
                            except ValueError:
                                pass
                        elif key in ['fps', 'framerate']:
                            try:
                                metadata_info['frame_rate'] = float(value)
                            except ValueError:
                                pass
                
            except Exception as e:
                print(f"Error reading metadata file {file_path}: {e}")
                continue
        
        return metadata_info
    
    def get_test_throw_data(self, test_throw_name: str) -> Optional[Dict]:
        """
        Get complete data for a specific test throw
        
        Args:
            test_throw_name: Name of the test throw folder
            
        Returns:
            Dictionary with all test throw data
        """
        test_throws = self.get_available_test_throws()
        
        for throw in test_throws:
            if throw['name'] == test_throw_name:
                return throw
        
        return None
    
    def calculate_frame_range(self, release_point: int, range_size: int = 30) -> Tuple[int, int]:
        """
        Calculate the frame range for analysis based on release point
        
        Args:
            release_point: Frame number of the release point
            range_size: Number of frames to analyze (default 30)
            
        Returns:
            Tuple of (start_frame, end_frame)
        """
        # Default: 30 frames before release point
        start_frame = max(1, release_point - range_size + 1)
        end_frame = release_point
        
        # If we don't have enough frames before release point, 
        # adjust to include some frames after
        if start_frame == 1 and release_point < range_size:
            end_frame = min(release_point + (range_size - release_point), release_point + 5)
        
        return start_frame, end_frame
    
    def calculate_default_frame_range(self, total_frames: int, frames_to_analyze: int = 30, frames_from_end: int = 50) -> Tuple[int, int]:
        """
        Calculate frame range when no release point is specified
        Analyzes 30 frames before the last 50 frames
        
        Args:
            total_frames: Total number of frames in the video
            frames_to_analyze: Number of frames to analyze (default 30)
            frames_from_end: Number of frames from the end to exclude (default 50)
            
        Returns:
            Tuple of (start_frame, end_frame)
            
        Example:
            If video has 760 frames total, analyze frames 680 to 710
            (30 frames before the last 50 frames)
        """
        if total_frames <= frames_from_end + frames_to_analyze:
            # If video is too short, analyze the middle portion
            start_frame = max(1, total_frames // 4)
            end_frame = min(total_frames, start_frame + frames_to_analyze)
        else:
            # Calculate: 30 frames before the last 50 frames
            end_frame = total_frames - frames_from_end  # Frame 710 for 760 total
            start_frame = max(1, end_frame - frames_to_analyze + 1)  # Frame 681 for 30 frames (681-710)
        
        return start_frame, end_frame
    
    def validate_test_throw_folder(self, folder_path: str) -> Dict:
        """
        Validate if a folder contains all necessary files for analysis
        
        Args:
            folder_path: Path to the test throw folder
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        if not os.path.exists(folder_path):
            validation['errors'].append("Folder does not exist")
            return validation
        
        # Check for video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
            video_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if len(video_files) < 2:
            validation['errors'].append("Need at least 2 video files (side and back views)")
        
        # Check for calibration files
        calib_extensions = ['*.csv', '*.txt']
        calibration_files = []
        for ext in calib_extensions:
            calibration_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Filter out metadata files
        actual_calibration_files = [f for f in calibration_files 
                                  if not any(keyword in os.path.basename(f).lower() 
                                           for keyword in ['metadata', 'info', 'release', 'frame'])]
        
        if len(actual_calibration_files) < 1:
            validation['errors'].append("Need at least 1 calibration file (.csv or .txt)")
        
        # Check for metadata
        metadata_files = [f for f in calibration_files 
                         if any(keyword in os.path.basename(f).lower() 
                              for keyword in ['metadata', 'info', 'release', 'frame'])]
        
        if len(metadata_files) == 0:
            validation['warnings'].append("No metadata file found - release point and frame rate will need to be entered manually")
        
        if len(validation['errors']) == 0:
            validation['valid'] = True
        
        return validation
