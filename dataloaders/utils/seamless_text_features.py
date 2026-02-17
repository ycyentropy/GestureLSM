import json
import numpy as np
import os
from loguru import logger


def process_word_data(data_dir, word_file, args, data, f_name, selected_file, lang_model):
    """Process word/text data from seamless dataset JSON files.
    
    Args:
        data_dir: Directory to save processed data
        word_file: Path to the JSON file containing word data
        args: Command line arguments, expected to contain pose_fps and t_pre_encoder
        data: Dictionary containing pose data, will be updated with word data
        f_name: File name identifier
        selected_file: DataFrame for selected files (not used but kept for compatibility)
        lang_model: Language model for word indexing
        
    Returns:
        Updated data dictionary with 'word' key containing processed word indices
    """
    logger.info(f"# ---- Building cache for Word {f_name} ---- #")

    # Check if word file exists
    if not os.path.exists(word_file):
        logger.warning(f"# ---- file not found for Word {f_name}, skip processing ---- #")
        return None

    try:
        # Load and parse JSON data
        with open(word_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"# ---- Failed to parse JSON file {word_file}: {e} ---- #")
        return None
    except Exception as e:
        logger.error(f"# ---- Failed to read JSON file {word_file}: {e} ---- #")
        return None

    # Extract word data and transcript time ranges from JSON
    all_words = []
    transcript_ranges = []
    for transcript in json_data.get('metadata:transcript', []):
        transcript_start = transcript.get('start')
        transcript_end = transcript.get('end')
        if transcript_start is not None and transcript_end is not None:
            transcript_ranges.append((transcript_start, transcript_end))
        all_words.extend(transcript.get('words', []))
    
    # Sort words by start time for easier processing
    # Handle None values by using a default value (0) for sorting
    all_words.sort(key=lambda x: x.get('start') if x.get('start') is not None else 0)
    
    # Sort transcript ranges by start time
    transcript_ranges.sort(key=lambda x: x[0])
    
    # Process basic encoding - map words to indices based on pose frames
    word_data, word_valid_data = process_basic_encoding(all_words, transcript_ranges, data, args, lang_model)

    # Update data with processed word indices and validity flags
    data['word'] = np.array(word_data)
    data['word_valid'] = np.array(word_valid_data)
    
    logger.info(f"# ---- Processed Word {f_name} ---- #")
    return data


def process_basic_encoding(word_list, transcript_ranges, data, args, lang_model):
    """Process basic word encoding for seamless dataset.
    
    Args:
        word_list: List of word dictionaries with start, end, and word keys
        transcript_ranges: List of (start, end) tuples for transcript time ranges
        data: Dictionary containing pose data for timing reference
        args: Command line arguments with pose_fps
        lang_model: Language model for word indexing
        
    Returns:
        Tuple of (word_indices, word_valid_flags) aligned with pose frames
    """
    word_data = []
    valid_data = []
    
    # Get total number of pose frames
    num_frames = data['pose'].shape[0]
    
    # Use two-pointer technique for O(n+m) complexity
    # current_time increases with fixed step size (1/pose_fps)
    transcript_idx = 0
    word_idx = 0
    num_transcripts = len(transcript_ranges)
    num_words = len(word_list)
    
    # Parameter to reduce PAD data by extending word time range
    time_extension = 0.1  # Extend word time range by 80ms to cover small gaps
    
    # Iterate through each pose frame
    for frame_idx in range(num_frames):
        # Calculate current time in seconds for this frame
        current_time = frame_idx / args.pose_fps
        
        # Check if current time is within any transcript's time range
        # Use two-pointer technique since current_time increases monotonically
        in_transcript = False
        while transcript_idx < num_transcripts and current_time > transcript_ranges[transcript_idx][1]:
            transcript_idx += 1
        
        if transcript_idx < num_transcripts and transcript_ranges[transcript_idx][0] <= current_time <= transcript_ranges[transcript_idx][1]:
            in_transcript = True
        
        # Add validity flag for this frame
        valid_data.append(in_transcript)
        
        # Find the corresponding word for this time
        # Use two-pointer technique since current_time increases monotonically
        found_word = False
        while word_idx < num_words:
            start_time = word_list[word_idx].get('start')
            end_time = word_list[word_idx].get('end')
            
            # Skip if either start or end time is None
            if start_time is None or end_time is None:
                word_idx += 1
                continue
            
            # If current_time is before this word, no need to check further words
            if current_time < start_time - time_extension:
                break
            
            # Check if current_time is within this word's time range (with extension) 是否会越界以及交叉区间？
            if start_time - time_extension <= current_time <= end_time + time_extension:
                word = word_list[word_idx].get('word', '')
                if word.strip():
                    word_data.append(lang_model.get_word_index(word))
                else:
                    word_data.append(lang_model.PAD_token)
                found_word = True
                break
            else:
                # current_time is after this word, move to next word
                word_idx += 1
        
        # If no word found for this time, use PAD token
        if not found_word:
            word_data.append(lang_model.PAD_token)
    
    return word_data, valid_data