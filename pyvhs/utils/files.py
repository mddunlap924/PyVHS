from pathlib import Path


class VideosToEdit:
    def __init__(self, path: str):
        self.path = Path(path)
        self.is_file = self.path.is_file()
        self.is_dir = self.path.is_dir()

        # Attributes to calculate
        self.original = list
        self.edited = list


    def list_videos(self):

        # List all files in directory
        if self.is_dir:
            # Assume only video files in the directory
            tmp = list(self.path.glob('**/*'))

            # Check that the path is a file
            tmp = [i for i in tmp if i.is_file()]

            # Remove text files from being processed if they are present
            tmp = [i for i in tmp if i.suffix != '.txt']

            # Video files to be processed
            self.original = tmp
        elif self.is_file:
            # Assume the file is video
            self.original = [self.path]
        else:
            raise ValueError('-dir provided is not a directory or file')

        # List edited files to be saved
        edited = []
        for file in self.original:
            name = file.stem + '_edited' + file.suffix
            edited.append(file.parent / name)
        self.edited = edited
        return
