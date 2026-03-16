python 02_data_sets/packopening/code/pipeline/03_extract_frames.py --all;python 02_data_sets/packopening/code/pipeline/03_extract_frames.py --all;python 02_data_sets/packopening/code/pipeline/03_extract_frames.py --all;
python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;

python 02_data_sets/packopening/code/pipeline/04_match_frames.py --video-id RzS1p0IWGH4 --rebuild;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all; python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all

python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all;
sqlite3 /Volumes/carbonite/claw/data/ccg_card_id/datasets/packopening/packopening.db "UPDATE videos SET status='downloaded' WHERE status='frames_extracted'"


sqlite3 /Volumes/carbonite/claw/data/ccg_card_id//datasets/packopening/packopening.db "UPDATE videos SET status='downloaded' WHERE status='frames_extracted'" a