from pathlib import Path
from itertools import groupby
import math
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import VideoFileClip, concatenate_videoclips


def convert(seconds):
    """
    Converts seconds into hours, minutes, seconds

    Args:
        seconds (_type_): Total seconds

    Returns:
        _type_: Hours, mintues, seconds
    """
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return '%dhours %02dminutes %02dseconds' % (hour, min, sec)


def templates_similarity(template_imgs: list,
                         img: np.array,
                         threshold: float):
    """
    The maximum Structural Similarity Index (SSIM) between a list of template images
    and a video frame.

    Args:
        template_imgs (list): List of images that will be identified and removed (templates)
        img (np.array): Frame of video
        threshold (float): SSIM threshold for similarity matching

    Returns:
        _type_: float, bool
    """
    # Maximum similarity metric for all templates
    similarity = []
    for template_img in template_imgs:
        sim = []
        for channel in [0, 1, 2]:
            temp_c = template_img[:, :, channel].squeeze()
            frame_c = img[:, :, channel].squeeze()
            sim.append(ssim(temp_c, frame_c))
        similarity.append(np.mean(sim))
    similarity = max(similarity)

    # Set threshold for similarity (is it the same as template)
    if similarity <= threshold:
        is_template = False
    else:
        is_template = True
    return similarity, is_template


def find_blanks(templates: list,
                clip_path: Path,
                threshold: float,
                timestamp: int):
    """
    Identify blank template frames from a video clip.
    This function can be passed into a Python multiprocessor for utilizing 
    all CPU cores.

    Args:
        templates (list): Template images to match
        clip_path (Path): Path to video
        threshold (float): SSIM threshold to identify a match
        timestamp (int): Timestamp to extract from video

    Returns:
        _type_: SSIM Score, Boolean for if a frame matches a template
    """
    clip_tmp = VideoFileClip(clip_path.as_posix())
    return templates_similarity(template_imgs=templates,
                                img=clip_tmp.get_frame(t=timestamp),
                                threshold=threshold
                                )


def extract_segments(blanks: list,
                     *,
                     num_sequentially_blanks: int=1,
                     num_sequentially_footage: int=1):
    """
    Extract footage and blank segments from a video
    
    https://stackoverflow.com/questions/61764744/
    efficient-ways-to-find-start-and-end-indexes-for-sequences
    
    Args:
        blanks (list): _description_
        num_sequentially_blanks (int, optional): 
        Number of blank frames that must occur sequentially. Defaults to 2.
        num_sequentially_footage (int, optional): 
        Number of footage frames that must occur sequentially. Defaults to 2.

    Returns:
        _type_: Blank segments, Footage segments
    """

    # Find sequences and lengths
    # seqs = [(key, length), ...]
    seqs = [(key, len(list(val))) for key, val in groupby(blanks)]

    # Find start positions of sequences
    # seqs = [(key, start, length), ...]
    seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]

    # Sequences equal to one
    # (i.e., no time-to-time differences in being identified as blank image)
    # index of blank segments in vec >= num_seq_blanks
    blank_segs = [[s[1], s[1] + s[2] - 1]
                  for s in seqs
                  if s[0] == 1 and s[2] >= num_sequentially_blanks]

    footage_segs = [[s[1], s[1] + s[2] - 1]
                    for s in seqs
                    if s[0] == 0 and s[2] >= num_sequentially_footage]

    # Add an time step at start and end to ensure no footage is lost
    # This will retain some blue screens
    footage_segs_final = []
    for footage_seg in footage_segs:
        start, end = footage_seg[0], footage_seg[1]
        if start == 0:
            start = 0
        else:
            start = start -1
        if end == len(blanks):
            end = end
        else:
            end = end + 1
        footage_segs_final.append([start, end])
    return blank_segs, footage_segs_final


class EditVideo:
    def __init__(self,
                 path_original: Path,
                 path_edited: Path,
                 templates: list,
                 *,
                 threshold: float=0.90,
                 interval: int=1):
        """
        Edit a video clip and remove blank segments

        Args:
            path_original (Path): Path to video to be edited
            path_edited (Path): Path to save the edited video
            templates (list): Template images 
            threshold (float, optional): SSIM threshold. Defaults to 0.90.
            interval (int, optional): Time interval to evaluate for blank frames [seconds].
            Defaults to 1.
        """
        self.path_original = path_original
        self.path_edited = path_edited
        self.templates = templates
        self.threshold = threshold

        # time interval to check frames [seconds]
        self.interval = interval

        # Video duration, number of frames, end time
        clip = VideoFileClip(self.path_original.as_posix())
        self.duration = clip.duration
        # self.nframes = clip.nframes
        self.end_time = math.floor(self.duration)
        del clip
        _ = gc.collect()

        # Path to save log file
        self.path_log = self.path_original.parent / (self.path_original.stem + '_log.txt')

        # Attributes to calculate
        self.blank_segments = []
        self.keep_segments = []
        return

    def __resize_templates(self, frame_w: int, frame_h: int):
        """
        Resize the template images to match the video frame size

        Args:
            frame_w (int): Template image width
            frame_h (int): Template image height

        Returns:
            _type_: Resized template images to match the video frames
        """
        templates_resized = []
        for template in self.templates:
            if template.ndim == 3:
                tmp = np.resize(template, (frame_h, frame_w, 3))
            else:
                tmp = np.resize(template, (frame_h, frame_w))
            templates_resized.append(tmp)
        return templates_resized


    def remove_blank_frames(self):
        """
        Identify and remove the blank frame

        Raises:
            AssertionError: No footage found in the video (an entirely blank video)

        Returns:
            _type_: Final edited video clip
        """
        # Load the clip
        clip = VideoFileClip(self.path_original.as_posix())

        # Resize template images to match the video's frame size
        templates = self.__resize_templates(frame_w=clip.w,
                                            frame_h=clip.h)

        # Timestamps in video to check for blank frame
        timestamps = list(range(0, self.end_time, self.interval))

        # Set to maximum CPU count
        pool = Pool(cpu_count())

        # Find video segments with footage to keep (use multiprocessing)
        func = partial(find_blanks,
                       templates,
                       self.path_original,
                       self.threshold)
        footage_blank_segments = pool.map(func, timestamps)
        pool.close()

        # Timestamps with blank frames
        blank_times = [int(i[1]) for i in footage_blank_segments]

        # Extract time segments with blanks (i.e., cutout)
        # and footage (i.e., keep)
        (blank_segments,
         keep_segments) = extract_segments(blanks=blank_times,
                                           num_sequentially_blanks=2,
                                           num_sequentially_footage=1)
        self.blank_segments = blank_segments
        self.keep_segments = keep_segments
        # Exit the program if no keep_segments are found (i.e., entire video is blank)
        if not keep_segments:
            raise AssertionError((f'No Footage Found for Video: '
                                  f'{self.path_original.as_posix()}'))

        # Concatenate keep segments into a final clip
        keep_clips = []
        for keep in keep_segments:
            keep_clips.append(clip.subclip(t_start=keep[0],
                                           t_end=keep[1]))
        final_clip = concatenate_videoclips(keep_clips)
        return final_clip


    def save_video(self, final_video_clip: VideoFileClip):
        """
        Save the final video clip to disk

        Args:
            final_video_clip (VideoFileClip): Final video clip to be saved to disk
        """
        # Save final video to disk
        final_video_clip.write_videofile(self.path_edited.as_posix(),
                                         verbose=False)
        return


    def save_logger(self,
                    final_video_clip,
                    *,
                    print_blanks: bool=True,
                    print_keeps: bool=False):
        """
        Save a text file showing the segments that were blank (i.e., removed). 
        The units are in seconds.

        Args:
            final_video_clip (VideoFileClip): Final video clip.
            print_blanks (bool, optional): Blank segments in the video clip. Defaults to True.
            print_keeps (bool, optional): Footage segments in the video clip. Defaults to False.
        """

        # Save log file to disk
        with open(self.path_log, 'w') as f:
            # Original Video Info:
            f.write((f'Original Video Information:\n\t'
                        f'Path: {self.path_original.as_posix()}'))
            f.write(f'Duration: {convert(seconds=self.duration)}\n')

            # Final Video Info:
            f.write((f'\nEdited/Saved Video Information:\n\t'
                        f'Path: {self.path_edited.as_posix()}'))
            f.write(f'Duration: {convert(seconds=final_video_clip.duration)}\n')

            # Amount of time cropped out of final video
            f.write((f'\nTotal Time of Blank Segments Removed: '
                        f'{convert(seconds=self.duration - final_video_clip.duration)}\n'))

            # Blank Segments
            if print_blanks:
                f.write('\nOriginal Video - Blank Segment Information')
                if self.blank_segments:
                    f.write(f'\n\tTotal Num. of Blank Segments: {len(self.blank_segments):,}\n')
                    for i, blank_seg in enumerate(self.blank_segments):
                        f.write((f'\tBlank Segment #{(i + 1):,}: '
                                 f'{blank_seg[0]:,} - {blank_seg[1]:,} seconds\n'))
                else:
                    f.write('\n\tNo Blank Segments Identified\n')

            # Keep Segments (i.e., footage saved in new video)
            if print_keeps:
                f.write('\nOriginal Video - Footage Segment Information')
                if self.keep_segments:
                    f.write(f'\n\tTotal Num. of Footage Segments: {len(self.keep_segments):,}\n')
                    for i, keep_seg in enumerate(self.keep_segments):
                        f.write((f'\tFootage Segment #{(i + 1):,}: '
                                 f'{keep_seg[0]:,} - {keep_seg[1]:,} seconds\n'))
                else:
                    f.write('\tNo Footage Segments Identified\n')
        print(f'\n\tLog File Saved at: \n\t{self.path_log}')
        return
    