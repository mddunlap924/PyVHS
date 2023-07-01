"""
Remove blank segments from video files located in a directory.
Author: Myles Dunlap
"""

# Libraries
import sys
import argparse
import gc
from pathlib import Path
import pkg_resources
import time
from PIL import Image
import numpy as np

# Custom modules
from pyvhs.utils.files import VideosToEdit
from pyvhs.utils.edits import EditVideo


# Detect debugging mode
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

# Packaged template images with PyVHS
DEFAULT_PATH_TEMPLATES = pkg_resources.resource_filename('pyvhs', 'template_imgs/')


def main():
    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = '/nvme4tb/test_video'
        args.template_imgs = DEFAULT_PATH_TEMPLATES
    else:
        arg_desc = '''Remove template images from a video file'''
        parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                         description= arg_desc)
        parser.add_argument("-dir",
                            required=True,
                            type=str,
                            help = "Directory containing video files")
        parser.add_argument("-template_imgs",
                            required=False,
                            default=DEFAULT_PATH_TEMPLATES,
                            type=str,
                            help = "Path to template images")
        parser.add_argument("-threshold",
                            required=False,
                            default=0.9,
                            type=float,
                            help = ('SSIM threshold for indicating a video frame '
                                    'as equivalent to a blank template. Default is 0.9.'))
        parser.add_argument("-time_interval",
                            required=False,
                            default=1,
                            type=int,
                            help = ('The time interval that video frames will be pulled and '
                                    'compared to template images. Default is 1 second.'))
        args = parser.parse_args()

    # List video files
    videos = VideosToEdit(path=args.dir)
    videos.list_videos()

    # Total Number of Files
    total_videos = len(videos.original)

    # Load templates
    template_imgs = []
    for f in Path(args.template_imgs).glob('*.png'):
        template_imgs.append(np.array(Image.open(f)))

    # Edit each video
    for i, (path_org, path_edit) in enumerate(zip(videos.original, videos.edited)):
        try:
            # Create a video editing object
            st = time.time()
            print(f'\nStarting Video {i + 1} of {total_videos}:\n\t{path_org}')
            video_edit = EditVideo(path_original=path_org,
                                   path_edited=path_edit,
                                   templates=template_imgs,
                                )

            # Identify segments of footage to keep
            final_clip = video_edit.remove_blank_frames()
            print((f'\nIdentified Blank Segments: '
                   f'{round((time.time() - st) / 60, 3)} mins.'))

            # Save edited video to disk
            st = time.time()
            if video_edit.blank_segments:
                print(f'Saving New Video at: {path_edit}')
                video_edit.save_video(final_video_clip=final_clip)
                print((f'\nEdited Video Saved: '
                    f'{round((time.time() - st) / 60, 3)} mins.'))
            else:
                print((f'Not Saving Edited Video because No Blank '
                       f'Segments Identified for Video.: \n\t'
                       f'{path_org.as_posix()}'))

            # Log which segments of the video were blank (i.e., removed and/or kept)
            video_edit.save_logger(final_video_clip=final_clip,
                                   print_blanks=True,
                                   print_keeps=True)

            # Clean up memory
            del video_edit, final_clip
            _ = gc.collect()
            print(f'Completed Video {i + 1} of {total_videos}\n\n')
        except Exception:
            print(f'Error for Video: {path_org.as_posix()}')
    print('PyVHS Completed - ENJOY YOUR VIDEOS!')
    return


if __name__ == "__main__":
    # Run main
    main()
